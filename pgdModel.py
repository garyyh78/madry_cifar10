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


class pgdModel:
    offSet = 50

    def __init__(self, model, epsilon, num_steps, step_size, random_start, loss_func):

        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.rand = random_start

        if loss_func == 'xent':
            loss = model.xent
        elif loss_func == 'cw':
            label_mask = tf.one_hot(model.y_input, Model.CIFAR_CLASS)
            pLogit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
            nLogit = tf.reduce_max((1 - label_mask) * model.pre_softmax- 1e4*label_mask, axis=1)
            loss = -tf.nn.relu(pLogit - nLogit + pgdModel.offSet)
        else:
            print('Unknown loss function. Defaulting to cross-entropy')
            loss = model.xent

        self.loss = loss
        self.grad = tf.gradients(loss, model.x_input)[0]
        self.acc = model.accuracy

    def baseline_perturb(self, sess, x_nat, y):

        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
        else:
            x = np.copy(x_nat)

        for i in range(self.num_steps):

            grad, loss, acc = sess.run([self.grad, self.loss, self.acc],
                                       feed_dict={self.model.x_input: x, self.model.y_input: y})

            x = np.add(x, self.step_size * np.sign(grad), out=x, casting='unsafe')
            x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
            x = np.clip(x, 0, 255)

            if (i % 2 == 0):
                print("steps %d, loss = %.6f, acc = %.4f" % (i, loss, acc))

        return x

    def _print_time(self):
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        print(st)

    # about 75%
    # simply trying to use a different starting point
    # probably should not use the adv model to train
    def perturb_v2(self, sess, x_nat, y):

        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon * 5, self.epsilon * 5, x_nat.shape)
        else:
            x = np.copy(x_nat)

        self._print_time()

        for i in range(self.num_steps):

            if i < 3:
                beta = 2.0
            elif i < 6:
                beta = 1.5
            else:
                beta = 1.0

            grad, loss, acc = sess.run([self.grad, self.loss, self.acc],
                                       feed_dict={self.model.x_input: x, self.model.y_input: y})

            x = np.add(x, beta * self.step_size * np.sign(grad), out=x, casting='unsafe')
            x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
            x = np.clip(x, 0, 255)

            if (i % 2 == 0):
                print("steps %d, loss = %.6f, acc = %.4f" % (i, loss, acc))

        return x

    # about 85%
    # maybe we indeed should start with advModel not natModel ??
    def perturb_v3(self, sess, x_nat, y):

        # starting point
        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
        else:
            x = np.copy(x_nat)

        self._print_time()

        # construct semi model
        semi_batch = 2
        semi_batch_size = self.num_steps // 2

        for b in range(semi_batch):

            # don't use true label, use some random label
            y_rand = np.copy(y)
            np.random.shuffle(y_rand)

            for i in range(semi_batch_size):
                acc, loss, grad = sess.run([self.acc, self.loss, self.grad],
                                           feed_dict={self.model.x_input: x, self.model.y_input: y_rand})

                # minimize so negative grad
                x = np.add(x, -self.step_size * np.sign(grad), out=x, casting='unsafe')

                # project back to feasible region
                d = np.subtract(x, x_nat, casting='unsafe')
                d_max = np.amax(np.absolute(d))
                d = np.multiply(d, self.epsilon / d_max, out=d)

                # clip
                x = np.add(x_nat, d, out=x, casting='unsafe')
                x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
                x = np.clip(x, 0, 255)

                print("steps %d : %d, loss = %.2f, acc = %.2f" % (b, i, loss, acc))

        loss, acc = sess.run([self.loss, self.acc],
                             feed_dict={self.model.x_input: x, self.model.y_input: y})

        print("final: loss = %.2f, acc = %.2f" % (loss, acc))

        return x

    # some research code
    def perturb(self, sess, x_nat, y):

        totalIter = 5
        maxeps = 5
        step_size = 2

        for j in range(maxeps):

            eps = 2 * j + 1

            # starting point
            if self.rand:
                x = x_nat + np.random.uniform(-eps, eps, x_nat.shape)
            else:
                x = np.copy(x_nat)

            self._print_time()

            for i in range(totalIter):
                # eval
                grad, loss, acc = sess.run([self.grad, self.loss, self.acc],
                                           feed_dict={self.model.x_input: x, self.model.y_input: y})
                print("%d/%d, loss = %.1f, acc = %.2f" % (eps, i, loss, acc))

                # update
                x = np.add(x, step_size * np.sign(grad), out=x, casting='unsafe')
                x = np.clip(x, x_nat - eps, x_nat + eps)
                x = np.clip(x, 0, 255)

        return x


with open('config.json') as config_file:
    config = json.load(config_file)

# it is better to start with natural model
model_dir = config['nat_model_dir']
print(model_dir)

model_file = tf.train.latest_checkpoint(model_dir)

if model_file is None:
    print('No model found')
    sys.exit()
else:
    print(model_file)

# ugly debugging
'''
from tensorflow.python import pywrap_tensorflow
import os
checkpoint_path = os.path.join(model_dir, "checkpoint-70000")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    if (key.startswith("unit_2_0/")):
        print("tensor_name: ", key)
'''

model = Model(mode='eval')

attackModel = pgdModel(model,
                       config['epsilon'],
                       config['num_steps'],
                       config['step_size'],
                       config['random_start'],
                       config['loss_func']
                       )

saver = tf.train.Saver()
data_path = config['data_path']

print(data_path, "\n")
cifar = cifar10_input.CIFAR10Data(data_path)
print("cifar10 data loaded\n")

with tf.Session() as sess:
    saver.restore(sess, model_file)
    print("model restored completed")

    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']

    print("num_eval_examples = %d, eval_batch_size = %d" % (num_eval_examples, eval_batch_size))
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    print("num_batches = %d\n" % (num_batches))

    print('Iterating over {} batches'.format(num_batches))
    path = config['store_adv_path']

    for i in range(num_batches):

        x_adv = []  # adv accumulator

        bstart = i * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        filename = "%s.r1.%d.b%04d" % (path, eval_batch_size, i)
        file = filename + ".npy"

        if os.path.exists(file):
            print("%s is generated, skip..." % (file))
            continue;

        print("bStart = %d, bEnd = %d, bSize = %d" % (bstart, bend, bend - bstart))

        x_batch = cifar.eval_data.xs[bstart:bend, :]
        y_batch = cifar.eval_data.ys[bstart:bend]

        x_batch_adv = attackModel.perturb(sess, x_batch, y_batch)
        x_adv.append(x_batch_adv)

        # repeatedly save/overwrite, note x_adv is cumulative
        x_adv = np.concatenate(x_adv, axis=0)
        np.save(filename, x_adv)
        print("last batch = %d, saved to %s" % (i, filename))
