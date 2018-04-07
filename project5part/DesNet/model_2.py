# -*- coding: utf-8 -*-
import os
from glob import glob
import time
import tensorflow as tf
import numpy as np
import matplotlib 
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from datasets import *


class DesNet(object):
    def __init__(self, sess, flags):
        self.sess = sess
        self.batch_size = flags.batch_size

        self.data_path = os.path.join('./Image', flags.dataset_name)
        self.output_dir = flags.output_dir

        self.log_dir = os.path.join(self.output_dir, 'log')
        self.sample_dir = os.path.join(self.output_dir, 'sample')
        self.model_dir = os.path.join(self.output_dir, 'checkpoints')

        self.original_images = tf.placeholder(shape=[None, flags.image_size, flags.image_size, 3], dtype = tf.float32)
        self.synthesized_images = tf.placeholder(shape=[None, flags.image_size, flags.image_size, 3], dtype = tf.float32)

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
        with tf.variable_scope('des', reuse = reuse):
            conv1 = tf.layers.conv2d(inputs, filters=64, kernel_size = [4,4], strides = (2,2), padding = 'same', name = 'conv1')
            bn1 = tf.layers.batch_normalization(conv1, training = is_training, name = 'bn1')
            relu1 = tf.nn.relu(bn1)
            conv2 = tf.layers.conv2d(relu1, filters = 128, kernel_size=[2,2], strides = (1,1), padding = 'same', name = 'conv2')
            bn2 = tf.layers.batch_normalization(conv2, training = is_training, name = 'bn2')
            relu2 = tf.nn.relu(bn2)
            fc = tf.contrib.layers.flatten(relu2)
            output = tf.layers.dense(fc, 1, name = 'fc')

        return output


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
            #energy
            #syn_res = self.descriptor(samples, to_sz, is_training=True, reuse=True)
            syn_res = self.descriptor(samples, reuse = tf.AUTO_REUSE)
            #Y_gradient
            #u
            noise = tf.random_normal(shape=tf.shape(samples), name='noise')
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
        #output1
        #self.m_original = self.descriptor(self.original_images)
        self.m_original = self.descriptor(self.original_images, reuse = tf.AUTO_REUSE)
        #output2
        #self.m_synthesized = self.descriptor(self.synthesized_images, reuse = True)
        self.m_synthesized = self.descriptor(self.synthesized_images, reuse = tf.AUTO_REUSE)
        #self.loss
        self.train_loss = tf.subtract(tf.reduce_mean(self.m_synthesized), tf.reduce_mean(self.m_original))

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'des' in var.name]
        #optim
        self.train_op = tf.train.AdamOptimizer(flags.learning_rate, beta1=flags.beta1).minimize(self.train_loss, var_list=self.d_vars)
        #self.samples
        self.sampling_op = self.Langevin_sampling(self.original_images, flags)

    def train(self, flags):
        # Prepare training data, scale is [0, 255]
        train_data = DataSet(self.data_path, image_size=flags.image_size)
        train_data = train_data.to_range(0, 255)
        print(train_data.shape)

        saver = tf.train.Saver(max_to_keep=50)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        self.sess.graph.finalize()

        print(" Start training ...")

        ####################################################
        # Train the model here. Print the loss term at each
        # epoch to monitor the training process. You may use
        # save_images() in ./datasets.py to save images. At
        # each log_step, record model in self.model_dir,
        # synthesized images in self.sample_dir,
        # loss in self.log_dir (using writer).
        ####################################################
        loss_all = []
        mean_data = np.mean(train_data, axis=(1, 2), keepdims = True)
        mean_images = np.zeros(np.shape(train_data)) + mean_data
        save_images(mean_images, self.sample_dir +'/mean_iamge' + '.png', space = 0, mean_img = None)
        for epoch in range(0, flags.epoch):
            #Y_sample = self.sess.run([self.sampling_op], feed_dict = {self.synthesized_images: mean_images})[0]
            Y_sample = self.sess.run([self.sampling_op], feed_dict = {self.original_images: mean_images})[0]
            _, loss = self.sess.run([self.train_op, self.train_loss], feed_dict = {self.original_images: Y_sample, 
                                        self.synthesized_images: train_data})
            loss_all.append(loss)
            if not epoch%20:
                print('epoch = ', epoch, 'loss = ', loss)
                save_images(Y_sample, self.sample_dir + '/sythesized sample' + str(epoch) + '.png', space = 0, mean_img = None)
        plt.figure()
        plt.plot(loss_all)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Training loss over iterations')
        plt.show()
        print('Training done')









