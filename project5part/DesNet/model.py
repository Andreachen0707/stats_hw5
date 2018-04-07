# -*- coding: utf-8 -*-
import os
from glob import glob
import time
import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from ops import *
from datasets import *


class DesNet(object):
    def __init__(self, sess, flags):
        self.sess = sess
        self.batch_size = flags.batch_size

        self.num_epochs = flags.epoch

        self.data_path = os.path.join('./Image', flags.dataset_name)
        self.output_dir = flags.output_dir

        self.log_dir = os.path.join(self.output_dir, 'log')
        self.sample_dir = os.path.join(self.output_dir, 'sample')
        self.model_dir = os.path.join(self.output_dir, 'checkpoints')

        self.original_images = tf.placeholder(tf.float32, shape=[None, flags.image_size, flags.image_size, 3])
        self.synthesized_images = tf.placeholder(tf.float32, shape=[None, flags.image_size, flags.image_size, 3])

        self.d_dim = 64

        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

        self.build_model(flags)

    def descriptor(self, inputs, is_training=True, reuse=False):
        ####################################################
        # Define network structure for descriptor.
        # Recommended structure:
        # conv1: channel 64 kernel 4*4 stride 2
        # conv2: channel 128 kernel 2*2 stride 1
        # fc: channel output 1
        # conv1 - bn - relu - conv2 - bn - relu -fc
        ####################################################
        with tf.variable_scope("des", reuse=reuse):
            d_conv1_bn = leaky_relu(batch_norm(conv2d(inputs,self.d_dim,kernal=(4,4), name='d_conv1_bn'), train=is_training, name='d_bn_1'))
            d_conv2_bn = leaky_relu(batch_norm(conv2d(d_conv1_bn, self.d_dim * 2, kernal=(2,2),strides=(1,1),name='d_conv2_bn'), train=is_training, name='d_bn_2'))
            d_conv2_bn = tf.reshape(d_conv2_bn, [-1,32*32*128])

            d_fc1 = linear(d_conv2_bn, 1, 'd_linear')
            # d_fc1 =  tf.nn.sigmoid(d_fc1)
            print(d_fc1)
            # d_fc1 = tf.cast(d_fc1,tf.bool)

            return d_fc1


    def Langevin_sampling(self, samples, flags):
        ####################################################
        # Define Langevin dynamics sampling operation.
        # To define multiple sampling steps, you may use
        # tf.while_loop to define a loop on computation graph.
        # The return should be the updated samples.
        ####################################################
        def cond(i, samples):
            return tf.less(i, flags.T)

        def body(i, samples):
            noise = tf.random_normal(shape=tf.shape(samples), name='noise')
            # syn_res = self.descriptor(samples, to_sz, is_training=True, reuse=True)
            syn_res = self.descriptor(samples, reuse=True)
            grad = tf.gradients(syn_res, samples, name='grad_des')[0]
            samples = samples - 0.5 * flags.delta * flags.delta * (samples / (flags.ref_sig * flags.ref_sig) - grad)
            samples = samples + flags.delta * noise
            return tf.add(i, 1), samples

        i = tf.constant(0)
        i, samples = tf.while_loop(cond, body, [i, samples])

        return samples

    def build_model(self, flags):
        ####################################################
        # Define the learning process. Record the loss.
        ####################################################
        # self.original_images = tf.placeholder(tf.float32, shape=[None, flags.image_size, flags.image_size, 3])
        # self.synthesized_images = tf.placeholder(tf.float32, shape=[None, flags.image_size, flags.image_size, 3])
        self.m_original = self.descriptor(self.original_images)
        self.m_synthesized = self.descriptor(self.synthesized_images, reuse=True)
        self.train_loss = tf.subtract(tf.reduce_mean(
                self.m_synthesized), tf.reduce_mean(self.m_original))

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'des' in var.name]

        self.train_op = tf.train.AdamOptimizer(flags.learning_rate, beta1=flags.beta1).minimize(self.train_loss, var_list=self.d_vars)
        self.sampling_op = self.Langevin_sampling(self.synthesized_images, flags)

    def train(self, flags):
        # Prepare training data, scale is [0, 255]
        train_data = DataSet(self.data_path, image_size=flags.image_size)
        train_data = train_data.to_range(0, 255)

        self.saver = tf.train.Saver(max_to_keep=50)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        num_batches = int(math.ceil(len(train_data) / self.batch_size))
        # loss = tf.reduce_sum(tf.reshape(self.train_loss, (-1, 64 * 64 * 3)))
        model_name = "DesNet.model"
        tf.summary.scalar('Loss', self.train_loss)

        summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        self.sess.graph.finalize()

        could_load, checkpoint_counter = self.load(self.model_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        print(" Start training ...")
        loss_des = []
        ####################################################
        # Train the model here. Print the loss term at each
        # epoch to monitor the training process. You may use
        # save_images() in ./datasets.py to save images. At
        # each log_step, record model in self.model_dir,
        # synthesized images in self.sample_dir,
        # loss in self.log_dir (using writer).
        ####################################################
        counter = 1

        sample_z = np.random.normal(0, 1, size=(7, 7))
        sample_inputs = train_data[0:1, :]

        # batch_images = train_data[idx * self.batch_size:(idx + 1) * self.batch_size, :]
        # batch_z = np.random.normal(0, 1, (self.batch_size, 7))





        for epoch in range(self.num_epochs):
            print(epoch)

            train_data_up = np.zeros((7, 64, 64, 3))
            train_data_up[:, :, :, 0] = np.mean(train_data[:, :, :, 0], axis=(1, 2), keepdims=True)
            train_data_up[:, :, :, 1] = np.mean(train_data[:, :, :, 1], axis=(1, 2), keepdims=True)
            train_data_up[:, :, :, 2] = np.mean(train_data[:, :, :, 2], axis=(1, 2), keepdims=True)
            print('zout', train_data_up.shape)
            for idx in range(0, num_batches):
                z_out = self.sess.run(
                    [self.sampling_op],
                    feed_dict={
                        self.synthesized_images: train_data_up
                    })[0]  # Update the H

                _, d_Loss, summary_loss = self.sess.run(
                    [self.train_op, self.train_loss, summary_op],
                    feed_dict={
                        self.original_images: train_data,
                        self.synthesized_images: z_out
                    })  # Update the H
                # print(d_Loss)
                loss_des.append(d_Loss)

                if np.mod(counter, 30) == 1:
                    samples = self.sess.run(
                        self.sampling_op,
                        feed_dict={
                            self.synthesized_images: train_data_up
                        }
                    )
                    print(samples.shape)

                    self.writer.add_summary(summary_loss, counter)
                    self.saver.save(self.sess, os.path.join(self.model_dir, model_name), global_step=counter)

                    save_images(samples,
                                '{}/train_{:02d}_{:04d}.png'.format(self.sample_dir, epoch, idx))
                    # print("[Sample] g_loss: %.8f" % (g_Loss))
                    #######################################################
                #                   end of your code
                #######################################################
                counter += 1
        # self.interpo_z()
        self.plot_loss(loss_des)

    def plot_loss(self, loss):
        plt.figure()
        plt.plot(loss)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Training loss over epochs')
        plt.show()

    def interpo_z(self):
        interpo_z = []
        for i in range(0, 10):
            for j in range(0, 10):
                z_tmp = [i * 0.4 - 2, j * 0.4 - 2]
                interpo_z.append(z_tmp)
        interpo_z = np.array(interpo_z)
        generate_img = self.sess.run(self.sampling_op, feed_dict={self.synthesized_images: interpo_z})
        save_images(generate_img, self.sample_dir + '/generate' + '.png', space=0, mean_img=None)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0



