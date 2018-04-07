from __future__ import division

import os
import math
import numpy as np
import tensorflow as tf


from six.moves import xrange

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from ops import *
from datasets import *


class GenNet(object):
    def __init__(self, sess, config):
        self.sess = sess
        self.batch_size = config.batch_size
        self.image_size = config.image_size

        self.g_lr = config.g_lr
        self.beta1 = config.beta1
        self.delta = config.delta
        self.sigma = config.sigma
        self.sample_steps = config.sample_steps
        self.z_dim = config.z_dim

        self.num_epochs = config.num_epochs
        self.data_path = os.path.join(config.data_path, config.category)
        self.log_step = config.log_step
        self.output_dir = os.path.join(config.output_dir, config.category)

        self.log_dir = os.path.join(self.output_dir, 'log')
        self.sample_dir = os.path.join(self.output_dir, 'sample')
        self.model_dir = os.path.join(self.output_dir, 'checkpoints')

        #my parameters
        self.g_dim = 64

        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

        self.obs = tf.placeholder(shape=[None, self.image_size, self.image_size, 3], dtype=tf.float32)
        self.z = tf.placeholder(shape=[None, self.z_dim], dtype=tf.float32)
        self.build_model()

    def generator(self, inputs, reuse=False, is_training=True):
        ####################################################
        # Define the structure of generator, you may use the
        # generator structure of DCGAN. ops.py defines some
        # layers that you may use.
        ####################################################
        with tf.variable_scope("gen", reuse=reuse):

            def conv_out_size_same(size, stride):
                return int(math.ceil(float(size) / float(stride)))

            s = self.image_size
            s2 = conv_out_size_same(s, 2)
            s4 = conv_out_size_same(s2, 2)
            s8 = conv_out_size_same(s4, 2)
            s16 = conv_out_size_same(s8, 2)

            z_t = linear(inputs, self.g_dim * 8 * s16 * s16, 'g_linear')
            print(s, s2, s4, s8, s16)

            h0 = tf.reshape(z_t, [-1, s16, s16, self.g_dim * 8])
            print(h0.shape)
            h0 = leaky_relu(batch_norm(h0, name='g_bn_0', train=is_training))

            output1_shape = [self.batch_size, s8, s8, self.g_dim * 4]
            g_conv1 = convt2d(h0, output1_shape, name='g_dcon_1')
            g_conv1 = leaky_relu(batch_norm(g_conv1, train=is_training, name='g_bn_1'))

            output2_shape = [self.batch_size, s4, s4, self.g_dim*2]
            g_conv2 = convt2d(g_conv1, output2_shape, name='g_dcon_2')
            g_conv2 = leaky_relu(batch_norm(g_conv2, train=is_training, name='g_bn_2'))

            output3_shape = [self.batch_size, s2, s2, self.g_dim]
            g_conv3 = convt2d(g_conv2, output3_shape, name='g_dcon_3')
            g_conv3 = leaky_relu(batch_norm(g_conv3, train=is_training, name='g_bn_3'))

            output4_shape = [self.batch_size, s, s, 3]
            g_conv4 = convt2d(g_conv3, output4_shape, name='g_dcon_4')
            g_conv4 = tf.nn.tanh(batch_norm(g_conv4, train=is_training, name='g_bn_4'))
            # g_conv4 = batch_norm(g_conv4, train=is_training, name='g_bn_4')
            print(g_conv1.shape, g_conv2.shape, g_conv3.shape, g_conv4.shape)
            return g_conv4

    def langevin_dynamics(self, z):
        ####################################################
        # Define Langevin dynamics sampling operation.
        # To define multiple sampling steps, you may use
        # tf.while_loop to define a loop on computation graph.
        # The return should be the updated z.
        ####################################################
        def cond(i, z):
            return tf.less(i, self.sample_steps)

        def body(i, z):
            noise = tf.random_normal(shape=tf.shape(z), name='noise')
            gen_res = self.generator(z, reuse=True)
            gen_loss = tf.reduce_mean(1.0 / (2 * self.sigma * self.sigma) * tf.square(self.obs - gen_res), axis=0)
            grad = tf.gradients(gen_loss, z, name='grad_des')[0]
            z = z - 0.5 * self.delta * self.delta * (z + grad)
            z = z + self.delta * noise
            return tf.add(i, 1), z

        i = tf.constant(0)
        i, z = tf.while_loop(cond, body, [i, z])
        return z

    def build_model(self):
        ####################################################
        # Define the learning process. Record the loss.
        ####################################################
        self.g_res = self.generator(self.z)
        self.gen_loss = tf.reduce_mean(1.0 / (2 * self.sigma * self.sigma) * tf.square(self.obs - self.g_res), axis=0)


        t_vars = tf.trainable_variables()
        self.g_vars = [var for var in t_vars if 'gen' in var.name]


        self.infer_op = self.langevin_dynamics(self.z)
        self.train_op = tf.train.AdamOptimizer(self.g_lr, beta1=self.beta1).minimize(self.gen_loss, var_list=self.g_vars)

    def train(self):
        # Prepare training data
        train_data = DataSet(self.data_path, image_size=self.image_size)
        train_data = train_data.to_range(-1, 1)

        num_batches = int(math.ceil(len(train_data) / self.batch_size))
        loss = tf.reduce_sum(tf.reshape(self.gen_loss,(-1,64*64*3)))
        model_name = "DCGAN.model"
        tf.summary.scalar('Loss',loss)
        summary_op = tf.summary.merge_all()

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver(max_to_keep=50)
        writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        self.sess.graph.finalize()

        print('Start training ...')

        ####################################################
        # Train the model here. Print the loss term at each
        # epoch to monitor the training process. You may use
        # save_images() in ./datasets.py to save images. At
        # each log_step, record model in self.model_dir,
        # reconstructed images and synthesized images in
        # self.sample_dir, loss in self.log_dir (using writer).
        ####################################################
        counter = 1


        sample_inputs = train_data[0:1, :]

        # batch_images = train_data[idx * self.batch_size:(idx + 1) * self.batch_size, :]
        batch_z = np.random.normal(0, 1, (self.batch_size, self.z_dim))

        for epoch in xrange(self.num_epochs):
            print(epoch)

            for idx in xrange(0, num_batches):

                z_out, _,out_image,g_Loss,summary_loss = self.sess.run(
                    [self.infer_op,self.train_op,self.g_res,self.gen_loss,summary_op],
                                         feed_dict={
                                             self.obs: train_data,
                                             self.z: batch_z
                                         })  # Update the H
                # print(g_Loss)
                # gLoss = self.sess.run([self.train_op],
                #                          feed_dict={
                #                              # self.obs: batch_images,
                #                              self.z: z_out[0]
                #                          })  # Update the generator

                # self.loss_summary = tf.summary.scalar("loss", self.gen_loss)
                if np.mod(counter,self.log_step) == 1:
                    sample_z = np.random.normal(0, 1, size=(16, self.z_dim))
                    samples = self.sess.run(
                        self.g_res,
                        feed_dict={
                            self.z: sample_z
                        }
                    )

                    writer.add_summary(summary_loss,counter)
                    saver.save(self.sess, os.path.join(self.model_dir, model_name),global_step=counter)

                    save_images(out_image,
                                '{}/reconstructed_{:02d}_{:04d}.png'.format(self.sample_dir, epoch, idx))
                    save_images(samples,
                                '{}/trained_{:02d}_{:04d}.png'.format(self.sample_dir, epoch, idx))
                    # print("[Sample] g_loss: %.8f" % (g_Loss))
                    #######################################################
                #                   end of your code
                #######################################################
                counter += 1
        self.interpo_z()

    def plot_loss(self, loss):
        plt.figure()
        plt.plot(loss)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Training loss over epochs')
        plt.show()


    def interpo_z(self):
        interpo_z = []
        for i in range(0, 8):
            for j in range(0, 8):
                z_tmp = [i * 0.5 - 2, j * 0.5 - 2]
                interpo_z.append(z_tmp)
        interpo_z = np.array(interpo_z)
        generate_img = self.sess.run(self.g_res, feed_dict={self.z: interpo_z})
        save_images(generate_img, self.sample_dir + '/generate' + '.png', space=0, mean_img=None)

    # @property
    # def model_dir(self):
    #     return "{}_{}_{}_{}".format(
    #         self.dataset_name, self.batch_size,
    #         self.output_size, self.output_size)
    #
    # def save(self, checkpoint_dir, step):
    #     model_name = "GenNet.model"
    #     checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
    #
    #     if not os.path.exists(checkpoint_dir):
    #         os.makedirs(checkpoint_dir)
    #
    #     self.saver.save(self.sess,
    #                     os.path.join(checkpoint_dir, model_name),
    #                     global_step=step)
    #
    # def load(self, checkpoint_dir):
    #     import re
    #     print(" [*] Reading checkpoints...")
    #     checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
    #
    #     ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    #     if ckpt and ckpt.model_checkpoint_path:
    #         ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    #         self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
    #         counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
    #         print(" [*] Success to read {}".format(ckpt_name))
    #         return True, counter
    #     else:
    #         print(" [*] Failed to find a checkpoint")
    #         return False, 0


