from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import time

import numpy as np
import tensorflow as tf

from restnet import Model

# baseline model
class pgdModelBasic:
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
            nLogit = tf.reduce_max((1 - label_mask) * model.pre_softmax, axis=1)
            loss = -tf.nn.relu(pLogit - nLogit + pgdModelBasic.offSet)
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
