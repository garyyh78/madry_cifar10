from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class Model(object):
    '''
    The following variables are exposed:

        mean_xent
        weight_decay_loss
        accuracy
        x_input
        y_input

    '''

    # constants here for CIFAR10
    CIFAR_HEIGHT = 32
    CIFAR_WIDTH = 32
    CIFAR_DEPTH = 3
    CIFAR_CLASS = 10

    FILTER_SIZE = 3
    IN_FILTER_SIZE = 3
    OUT_FILTER_SIZE = 16

    # mode can be train or eval
    def __init__(self, mode):

        assert mode == 'train' or mode == 'eval'

        print("model run as %s\n" % (mode))

        self.mode = mode
        self._stride_one = self._stride_arr(1)

        self._build_model()

    # main graph
    def _build_model(self):

        with tf.variable_scope('input'):

            self.x_input = tf.placeholder(tf.float32,
                                          shape=[None, Model.CIFAR_HEIGHT, Model.CIFAR_WIDTH, Model.CIFAR_DEPTH])
            self.y_input = tf.placeholder(tf.int64)

            # zero mean and unit norm
            norm_input = tf.map_fn(lambda img: tf.image.per_image_standardization(img), self.x_input)

            x = self._conv('init_conv',
                           norm_input,
                           Model.FILTER_SIZE,
                           Model.IN_FILTER_SIZE,
                           Model.OUT_FILTER_SIZE,
                           self._stride_one)

        # W28-10, n = 4, l = 28, k = 10
        strides = [1, 2, 2]
        activate_before_residual = [True, False, False]
        N = 4  # layer = 4 * 2 * n + 4
        K = 10
        res_func = self._residual

        # https://arxiv.org/pdf/1605.07146v1.pdf
        filters = [16, 16 * K, 32 * K, 64 * K]

        for j in range(0, 3):
            layer = j + 1 # 1, 2, 3
            with tf.variable_scope('unit_%d_0' % layer):
                x = res_func(x,
                             filters[j],
                             filters[layer],
                             self._stride_arr(strides[j]),
                             activate_before_residual[j])

            for i in range(1, N): # 1, 2, .. 4( N-1 )
                with tf.variable_scope('unit_%d_%d' % (layer, i)):
                    x = res_func(x, filters[layer], filters[layer], self._stride_one, False)

        # output layer
        with tf.variable_scope('unit_last'):
            x = self._batch_norm('final_bn', x)
            x = self._relu(x, 0.1)
            x = self._global_avg_pool(x)

        with tf.variable_scope('logit'):
            self.pre_softmax = self._fully_connected(x, Model.CIFAR_CLASS)

        self.predictions = tf.argmax(self.pre_softmax, 1)
        self.correct_prediction = tf.equal(self.predictions, self.y_input)
        self.num_correct = tf.reduce_sum(tf.cast(self.correct_prediction, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        with tf.variable_scope('costs'):
            self.y_xent = \
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pre_softmax,
                                                               labels=self.y_input)
            self.xent = tf.reduce_sum(self.y_xent, name='y_xent')
            self.mean_xent = tf.reduce_mean(self.y_xent)
            self.weight_decay_loss = self._l2_decay()

    # ----- help layers/nodes below ------
    def _residual(self, x, in_filter, out_filter, stride, activate_before_residual=False):
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, 0.1)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, 0.1)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu(x, 0.1)
            x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
                orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0],
                                         [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]])
            x += orig_x

        tf.logging.debug('image after unit %s', x.get_shape())
        return x

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = \
                tf.get_variable('DW',
                                [filter_size, filter_size, in_filters, out_filters],
                                tf.float32,
                                initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))

        return tf.nn.conv2d(x, kernel, strides, padding='SAME')

    def _fully_connected(self, x, out_dim):

        num_non_batch_dimensions = len(x.shape)
        prod_non_batch_dimensions = 1

        for ii in range(num_non_batch_dimensions - 1):
            prod_non_batch_dimensions *= int(x.shape[ii + 1])

        x = tf.reshape(x, [tf.shape(x)[0], -1])

        w = tf.get_variable('DW',
                            [prod_non_batch_dimensions, out_dim],
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))

        b = tf.get_variable('biases', [out_dim], initializer=tf.constant_initializer())

        return tf.nn.xw_plus_b(x, w, b)

    def _l2_decay(self):

        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find('DW') > 0:
                costs.append(tf.nn.l2_loss(var))

        return tf.add_n(costs)

    def _stride_arr(self, stride):
        return [1, stride, stride, 1]

    def _batch_norm(self, name, x):
        with tf.name_scope(name):
            return tf.contrib.layers.batch_norm(inputs=x,
                                                decay=.9,
                                                center=True,
                                                scale=True,
                                                activation_fn=None,
                                                updates_collections=None,
                                                is_training=(self.mode == 'train')
                                                )

    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4  # NHWC
        return tf.reduce_mean(x, [1, 2])

    def _relu(self, x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')
