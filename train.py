from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import shutil
from datetime import datetime
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf

import cifar10_input
from pgdModelBasic import pgdModelBasic
from restnet import Model

with open('config.json') as config_file:
    config = json.load(config_file)

# seeding randomness
tf.set_random_seed(config['tf_random_seed'])
np.random.seed(config['np_random_seed'])

# Setting up training parameters
max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
step_size_schedule = config['step_size_schedule']
weight_decay = config['weight_decay']
data_path = config['data_path']
momentum = config['momentum']
batch_size = config['training_batch_size']

# Setting up the data and the model
raw_cifar = cifar10_input.CIFAR10Data(data_path)
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model(mode='train')

# Setting up the optimizer
boundaries = [int(sss[0]) for sss in step_size_schedule]
boundaries = boundaries[1:]

values = [sss[1] for sss in step_size_schedule]
learning_rate = tf.train.piecewise_constant(
    tf.cast(global_step, tf.int32),
    boundaries,
    values)

total_loss = model.mean_xent + weight_decay * model.weight_decay_loss
train_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(
    total_loss,
    global_step=global_step)

# Set up adversary model
attack = pgdModelBasic(model,
                       config['epsilon'],
                       4,  # num of steps
                       2,  # step size
                       config['random_start'],
                       config['loss_func'])

# Setting up the Tensorboard and checkpoint outputs
model_dir = config['my01_model_dir']

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    model_file = ""
else:
    model_file = tf.train.latest_checkpoint(model_dir)

# keep the configuration file with the model for reproducibility
shutil.copy('config.json', model_dir)
saver = tf.train.Saver(max_to_keep=1)

with tf.Session() as sess:
    if model_file:
        saver.restore(sess, model_file)
        print("model restore completed: " + model_file)
    else:
        sess.run(tf.global_variables_initializer())

    # initialize data augmentation
    cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, model)

    training_time = 0.0

    # Main training loop
    for ii in range(max_num_training_steps):
        x_batch, y_batch = cifar.train_data.get_next_batch(batch_size,
                                                           multiple_passes=True)

        # Compute Adversarial Perturbations
        start = timer()
        x_batch_adv = attack.baseline_perturb(sess, x_batch, y_batch)
        end = timer()
        training_time += end - start
        print("perturbed")

        nat_dict = {model.x_input: x_batch,
                    model.y_input: y_batch}

        adv_dict = {model.x_input: x_batch_adv,
                    model.y_input: y_batch}

        # Output to stdout
        if ii % num_output_steps == 0:

            nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
            adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)

            print('Step {}:    ({})'.format(ii, datetime.now()))
            print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
            print('    training adv accuracy {:.4}%'.format(adv_acc * 100))

            if ii != 0:
                print('    {} examples per second'.format(
                    num_output_steps * batch_size / training_time))
                training_time = 0.0

        # Write a checkpoint
        if ii % num_checkpoint_steps == 0:
            saver.save(sess,
                       os.path.join(model_dir, 'checkpoint'),
                       global_step=global_step)

        # Actual training step
        start = timer()
        sess.run(train_step, feed_dict=adv_dict)
        end = timer()
        training_time += end - start
        print("trained")
