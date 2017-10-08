from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import json
import math
import os
import sys
import time

import numpy as np
import tensorflow as tf

import cifar10_input
from restnet import Model


class pgdModelEn:
    offSet = 50

    def __init__(self,
                 sess_graph_file_models,
                 epsilon,
                 num_steps,
                 step_size,
                 random_start,
                 ):

        self.sess_graph_model_loss_grad_acc_list = []

        for sess, graph, _, model in sess_graph_file_models:
            loss = model.xent
            grad = tf.gradients(loss, model.x_input)[0]
            acc = model.accuracy

            self.sess_graph_model_loss_grad_acc_list.append((sess, graph, model, loss, grad, acc))

        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.rand = random_start

    def _print_time(self):
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        print(st)

    def perturb(self, x_nat, y):

        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
        else:
            x = np.copy(x_nat)

        self._print_time()

        prev_total_loss = -1
        prev_total_acc = -1

        for i in range(self.num_steps):

            # speed up
            if i > 5:
                size_mult = 1  # step 6 and above
            elif i > 2:
                size_mult = 2  # step 3, 4, 5
            else:
                size_mult = 4  # step 0, 1, 2

            # perturb twice
            total_loss = 0
            total_acc = 0
            for sess_, _, model_, loss_, grad_, acc_ in self.sess_graph_model_loss_grad_acc_list:
                with sess_.as_default():
                    grad, loss, acc = sess_.run([grad_, loss_, acc_],
                                                feed_dict={model_.x_input: x, model_.y_input: y}
                                                )

                    x = np.add(x, size_mult * self.step_size * np.sign(grad), out=x, casting='unsafe')
                    x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
                    x = np.clip(x, 0, 255)

                    print("steps %d, loss = %.1f, acc = %.2f" % (i, loss, acc))

                    total_loss += loss
                    total_acc += acc

            # check early stop conditions
            loss_delta = total_loss - prev_total_loss
            acc_delta = total_acc - prev_total_acc
            if prev_total_loss > 0 and loss_delta < 10 and prev_total_acc > 0 and acc_delta > -0.01:
                print("stop: prev loss = %.1f curr loss = %.1f prev acc = %.2f curr acc = %.2f" % (
                    prev_total_loss, total_loss, prev_total_acc, total_acc))
                break
            else:
                prev_total_loss = total_loss
                prev_total_acc = total_acc

        return x

# main configure file
with open('config.json') as config_file:
    config = json.load(config_file)

# Model prep
model_dirs = [config['nat_model_dir'], config['adv_model_dir']]
sess_graph_file_models = []

for path in model_dirs:

    print(path)
    model_file = tf.train.latest_checkpoint(path)

    if model_file is None:
        print('No model found')
        sys.exit()
    else:
        print(model_file)

    graph = tf.Graph()
    with graph.as_default():
        model = Model(mode='eval')
    sess = tf.Session(graph=graph)

    sess_graph_file_models.append((sess, graph, model_file, model))

attackModelEn = pgdModelEn(sess_graph_file_models,
                           config['epsilon'],
                           config['num_steps'],
                           config['step_size'],
                           config['random_start']
                           )

for sess, graph, file, _ in sess_graph_file_models:
    with sess.as_default():
        with graph.as_default():
            tf.global_variables_initializer().run()
            model_saver = tf.train.Saver(tf.global_variables())
            model_saver.restore(sess, file)

# Data prep
data_path = config['data_path']
print(data_path, "\n")
cifar = cifar10_input.CIFAR10Data(data_path)
print("cifar10 data loaded\n")

num_eval_examples = config['num_eval_examples']
eval_batch_size = config['eval_batch_size']

print("num_eval_examples = %d, eval_batch_size = %d" % (num_eval_examples, eval_batch_size))
num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
print("num_batches = %d\n" % (num_batches))

print('Iterating over {} batches'.format(num_batches))
path = config['store_adv_path']

# Run attack
for i in range(num_batches):

    x_adv = []  # adv accumulator

    bstart = i * eval_batch_size
    bend = min(bstart + eval_batch_size, num_eval_examples)

    filename = "%s.en.%d.b%04d" % (path, eval_batch_size, i)
    file = filename + ".npy"

    if os.path.exists(file):
        print("%s is generated, skip..." % (file))
        continue;

    print("bStart = %d, bEnd = %d, bSize = %d" % (bstart, bend, bend - bstart))

    x_batch = cifar.eval_data.xs[bstart:bend, :]
    y_batch = cifar.eval_data.ys[bstart:bend]

    x_batch_adv = attackModelEn.perturb(x_batch, y_batch)
    x_adv.append(x_batch_adv)

    # repeatedly save/overwrite, note x_adv is cumulative
    x_adv = np.concatenate(x_adv, axis=0)
    np.save(filename, x_adv)
    print("last batch = %d, saved to %s" % (i, filename))

# shutdown
for sess, _, _, _ in sess_graph_file_models:
    sess.close()
