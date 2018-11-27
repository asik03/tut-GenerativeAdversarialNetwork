import os
#import cv2
import glob
import gzip
import math
import itertools
import numpy as np
import pylab as plt
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# from PIL import Image
# from tqdm import tqdm
from six.moves import xrange
#from urllib.request import urlretrieve
#from scipy.misc import imsave, imread, imresize
from tensorflow.examples.tutorials.mnist import input_data

# comment below two lines to implement this code with GPU
#os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(DIR_PATH, "data")  # path for your results
NUM_IMAGES = 60000
DATABASE_NAME = 'mnist'
OUTPUT_NAME = 'out'
OUTPUT_PATH = os.path.join(DATA_DIR, OUTPUT_NAME)


# This section is to implement DCGAN. You need to complete TODO parts by yourself according to DCGAN topology
class DCGAN():
    def __init__(self, sess, img_size, z_dim, batch_size, epoch):
        self.sess = sess
        self.epoch = epoch
        self.z_dim = z_dim
        self.img_size = img_size
        self.img_dim = img_size * img_size
        self.img_shape = [img_size, img_size, 1]
        self.batch_size = batch_size

        self.build_model()
        self.model_name = "DCGAN.model"

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.img = tf.placeholder(tf.float32, [None] + self.img_shape, name='real_images')

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = tf.summary.histogram('z', self.z)

        initializer = tf.contrib.layers.xavier_initializer()

        # self.img_fake is produced by generator with a random input z.
        self.img_fake = self.generator(self.z)

        # Outputs from discriminator with real image or fake image.
        self.D, self.D_logits_real = self.discriminator(self.img)
        self.D_fake, self.D_logits_fake = self.discriminator(self.img_fake, reuse=True)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_fake)
        self.img_fake_sum = tf.summary.histogram("G", self.img_fake)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
        self.g_vars = [var for var in t_vars if var.name.startswith('generator')]

        # Calculating Loss value
        self.D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_real, labels=tf.ones_like(self.D_logits_real)))
        self.D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_fake,
                                                    labels=tf.zeros_like(self.D_logits_fake)))
        self.D_loss = self.D_loss_real + self.D_loss_fake
        self.G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_fake, labels=tf.ones_like(self.D_logits_fake)))

    def generator(self, z, is_training=True, reuse=False):
        # TODO: generate fake image with randonm z with 4 layers
        alpha = 0.2
        with tf.variable_scope("generator") as scope:
            # First fully connected layer
            x1 = tf.layers.dense(z, 7 * 7 * 256)
            x1 = tf.reshape(x1, (-1, 7, 7, 256))
            x1 = tf.layers.batch_normalization(x1, training=is_training)
            x1 = tf.maximum(alpha * x1, x1)  # works as relu

            # Second convolution layer
            # TODO: After 2d transpose convolution, the shape of x2 is 14*14*128
            x2 = tf.layers.conv2d_transpose(x1, 128, 5, strides=2, padding="same")
            # TODO: Batch normalization of x2
            # Reduces the internal covariance shift i.e. it makes the learning of layers in the network more independent of each other.
            x2 = tf.layers.batch_normalization(x2, training=is_training)
            # AS relu
            x2 = tf.maximum(alpha * x2, x2)

            # Third convolution layer
            # TODO:2d transpose convolution as the second convolution layer, after that, the shape of x3 is 28*28*64
            x3 = tf.layers.conv2d_transpose(x2, 64, 5, strides=2, padding="same")
            # Batch normalization of x3
            x3 = tf.layers.batch_normalization(x3, training=is_training)
            # As relu
            x3 = tf.maximum(alpha * x3, x3)
            # Dropout
            drop = tf.nn.dropout(x3, keep_prob=0.5)

            # Output layer
            logits = tf.layers.conv2d_transpose(drop, 1, 5, strides=1, padding="same")

            out = tf.tanh(logits)

            return out

    def discriminator(self, img, reuse=False):
        # TODO: discriminate the input img whether a real one or fake one.
        alpha = 0.2
        with tf.variable_scope("discriminator", reuse=reuse):
            # The shape of img is 28*28*3, the shape of x1 is 14*14*64
            x1 = tf.layers.conv2d(img, 64, 5, strides=2, padding='same')
            x1 = tf.maximum(alpha * x1, x1)

            # TODO: Second convolution layer, the shape of x2 is 7*7*128
            x2 = tf.layers.conv2d(x1, 128, 5, strides=2, padding='same')

            # TODO: Batch normalization of x2
            bn2 = tf.layers.batch_normalization(x2)
            x2 = tf.maximum(alpha * bn2, bn2)

            # TODO: complete third convolution layer, the shape of x3 should be 4*4*256
            x3 = tf.layers.conv2d(x2, 256, 5, strides=2, padding='same')
            # batch_normalization
            bn3 = tf.layers.batch_normalization(x3)
            x3 = tf.maximum(alpha * bn3, bn3)

            # last layer
            x4 = tf.reshape(x3, (-1, 4 * 4 * 256))
            logits = tf.layers.dense(x4, 1)
            out = tf.sigmoid(logits)
            return out, logits

    def plot(self, samples):
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(self.img_size, self.img_size), cmap="Greys_r")

        return fig

    def train(self, output_path):
        d_optim = tf.train.AdamOptimizer().minimize(self.D_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer().minimize(self.G_loss, var_list=self.g_vars)

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        i = 0
        counter = 1
        mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

        for epoch in xrange(self.epoch):
            input_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
            input_imgs, _ = mnist.train.next_batch(self.batch_size)
            input_imgs_ = input_imgs.reshape((self.batch_size, self.img_size, self.img_size, 1))
            # Update D network
            _, D_loss_curr = self.sess.run([d_optim, self.D_loss], feed_dict={self.img: input_imgs_, self.z: input_z})
            _, G_loss_curr = self.sess.run([g_optim, self.G_loss], feed_dict={self.z: input_z})

            t_vars = tf.trainable_variables()
            d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
            g_vars = [var for var in t_vars if var.name.startswith('generator')]

            counter += 1
            if counter % 1000 == 0:
                print("Epoch: [{:2d}] D_loss: {:.8f}, G_loss {:.8f}".format(
                    epoch, D_loss_curr, G_loss_curr))

                samples = self.sess.run(self.img_fake, feed_dict={self.z: input_z})
                # import pdb; pdb.set_trace()
                fig = self.plot(samples)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                plt.savefig(os.path.join(output_path, '{}.png'.format(str(i).zfill(3))), bbox_inches='tight')
                i += 1
                plt.close(fig)


def run():
    # save
    flags = tf.app.flags
    flags.DEFINE_integer("img_size", 28, "Image size.")
    flags.DEFINE_integer("z_dim", 100, "The dimension of random input z.")
    flags.DEFINE_integer("hidden_dim", 128, "The dimension of hidden ")
    flags.DEFINE_integer("batch_size", 16, "The size of batch images [32]")
    flags.DEFINE_integer("epoch", 100000, "Epoch to train!")
    FLAGS = flags.FLAGS

    config = tf.ConfigProto(
        device_count={'GPU': 1})  # If you wanna implement this code with GPU, change 0 to 1 or 2
    with tf.Session(config=config) as sess:
        dcgan = DCGAN(sess, img_size=FLAGS.img_size, z_dim=FLAGS.z_dim,
                      batch_size=FLAGS.batch_size, epoch=FLAGS.epoch)
        var = dcgan.img_fake
        print(var)
        dcgan.train(OUTPUT_PATH)


if __name__ == "__main__":
    run()
