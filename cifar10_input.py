from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import numpy as np
import tensorflow as tf


class CIFAR10Data(object):
    def __init__(self, path):

        train_filenames = ['data_batch_{}'.format(i + 1) for i in range(5)]
        eval_filename = 'test_batch'
        metadata_filename = 'batches.meta'

        train_images = np.zeros((50000, 32, 32, 3), dtype='uint8')
        train_labels = np.zeros(50000, dtype='int32')

        for i, fname in enumerate(train_filenames):
            cur_images, cur_labels = self._load_datafile(os.path.join(path, fname))
            train_images[i * 10000: (i + 1) * 10000, ...] = cur_images
            train_labels[i * 10000: (i + 1) * 10000, ...] = cur_labels

        eval_images, eval_labels = self._load_datafile(os.path.join(path, eval_filename))

        with open(os.path.join(path, metadata_filename), 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
            self.label_names = data_dict[b'label_names']

        for i in range(len(self.label_names)):
            self.label_names[i] = self.label_names[i].decode('utf-8')

        self.train_data = DataSubset(train_images, train_labels)
        self.eval_data = DataSubset(eval_images, eval_labels)

    @staticmethod
    def _load_datafile(filename):

        with open(filename, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
            assert data_dict[b'data'].dtype == np.uint8
            image_data = data_dict[b'data']
            image_data = image_data.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
            return image_data, np.array(data_dict[b'labels'])


class DataSubset(object):
    def __init__(self, xs, ys):
        self.xs = xs
        self.n = xs.shape[0]
        self.ys = ys
        self.batch_start = 0
        self.cur_order = np.random.permutation(self.n)

    def get_next_batch(self, batch_size, multiple_passes=False, reshuffle_after_pass=True):

        if self.n < batch_size:
            raise ValueError('Batch size can be at most the dataset size')

        if not multiple_passes:
            actual_batch_size = min(batch_size, self.n - self.batch_start)
            if actual_batch_size <= 0:
                raise ValueError('Pass through the dataset is complete.')

            batch_end = self.batch_start + actual_batch_size
            batch_xs = self.xs[self.cur_order[self.batch_start: batch_end], ...]
            batch_ys = self.ys[self.cur_order[self.batch_start: batch_end], ...]
            self.batch_start += actual_batch_size

            return batch_xs, batch_ys

        actual_batch_size = min(batch_size, self.n - self.batch_start)
        if actual_batch_size < batch_size:
            if reshuffle_after_pass:
                self.cur_order = np.random.permutation(self.n)

            self.batch_start = 0

        batch_end = self.batch_start + batch_size
        batch_xs = self.xs[self.cur_order[self.batch_start: batch_end], ...]
        batch_ys = self.ys[self.cur_order[self.batch_start: batch_end], ...]
        self.batch_start += actual_batch_size
        return batch_xs, batch_ys


class AugmentedDataSubset(object):
    def __init__(self, raw_datasubset, sess, x_input_placeholder, augmented):
        self.sess = sess
        self.raw_datasubset = raw_datasubset
        self.x_input_placeholder = x_input_placeholder
        self.augmented = augmented

    def get_next_batch(self, batch_size, multiple_passes=False, reshuffle_after_pass=True):
        raw_batch = self.raw_datasubset.get_next_batch(batch_size,
                                                       multiple_passes,
                                                       reshuffle_after_pass)

        images = raw_batch[0].astype(np.float32)

        return self.sess.run(self.augmented, feed_dict={self.x_input_placeholder: raw_batch[0]}), raw_batch[1]


class AugmentedCIFAR10Data(object):
    def __init__(self, raw_cifar10data, sess, model):
        assert isinstance(raw_cifar10data, CIFAR10Data)
        self.image_size = 32

        # create augmentation computational graph
        self.x_input_placeholder = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        padded = tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(img,
                                                                              self.image_size + 4,
                                                                              self.image_size + 4
                                                                              ),
                           self.x_input_placeholder)

        cropped = tf.map_fn(lambda img: tf.random_crop(img, [self.image_size,
                                                             self.image_size,
                                                             3]), padded)

        flipped = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), cropped)

        self.augmented = flipped
        self.train_data = AugmentedDataSubset(raw_cifar10data.train_data,
                                              sess,
                                              self.x_input_placeholder,
                                              self.augmented)

        self.eval_data = AugmentedDataSubset(raw_cifar10data.eval_data,
                                             sess,
                                             self.x_input_placeholder,
                                             self.augmented)

        self.label_names = raw_cifar10data.label_names
