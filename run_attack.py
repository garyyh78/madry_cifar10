from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math

import numpy as np
import tensorflow as tf

import cifar10_input
from restnet import Model


def run_attack(checkpoint, x_adv, epsilon):
    cifar = cifar10_input.CIFAR10Data(data_path)
    model = Model(mode='eval')
    saver = tf.train.Saver()

    num_eval_examples = adv_count
    eval_batch_size = 100

    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    total_corr = 0

    # another round of check
    x_nat = cifar.eval_data.xs[:adv_count, :, :, :]
    l_inf = np.amax(np.abs(x_nat - x_adv))
    if l_inf > epsilon + 0.0001:
        print('maximum perturbation found: {}'.format(l_inf))
        print('maximum perturbation allowed: {}'.format(epsilon))
        return

    y_pred = []  # label accumulator

    with tf.Session() as sess:

        saver.restore(sess, checkpoint)
        print("model restored")

        for ibatch in range(num_batches):
            bstart = ibatch * eval_batch_size
            bend = min(bstart + eval_batch_size, num_eval_examples)

            x_batch = x_adv[bstart:bend, :]
            y_batch = cifar.eval_data.ys[bstart:bend]

            print("bStart = %d, bEnd = %d, bSize = %d" % (bstart, bend, bend - bstart))
            dict_adv = {model.x_input: x_batch, model.y_input: y_batch}
            cur_corr, y_pred_batch = sess.run([model.num_correct, model.predictions],
                                              feed_dict=dict_adv)

            total_corr += cur_corr
            y_pred.append(y_pred_batch)

    accuracy = total_corr / num_eval_examples
    print('Accuracy: {:.2f}%'.format(100.0 * accuracy))

    y_pred = np.concatenate(y_pred, axis=0)
    np.save('pred.npy', y_pred)
    print('Output saved at pred.npy')


# main entry
with open('config.json') as config_file:
    config = json.load(config_file)

data_path = config['data_path']
model_dir = config['adv_model_dir']
print(data_path)
print(model_dir)
checkpoint = tf.train.latest_checkpoint(model_dir)
print(checkpoint)

attackFile = config['store_adv_path'] + ".bs100.b0001" + ".npy"

print(attackFile)
x_adv = np.load(attackFile)
adv_count = x_adv.shape[0]
print(adv_count)

# sanity checks
if checkpoint is None:
    print('No checkpoint found')
elif x_adv.shape != (adv_count, 32, 32, 3):
    print('Invalid shape: expected (10000, 32, 32, 3), found {}'.format(x_adv.shape))
elif np.amax(x_adv) > 255.0001 or np.amin(x_adv) < -0.0001:
    print('Invalid pixel range. Expected [0, 255], found [{}, {}]'.format(np.amin(x_adv), np.amax(x_adv)))
else:
    print("sanity check is all good")
    run_attack(checkpoint, x_adv, config['epsilon'])
