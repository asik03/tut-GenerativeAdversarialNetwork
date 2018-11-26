import os
#import cv2
import glob
import gzip
import itertools
import numpy as np
import pylab as plt
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#from PIL import Image
#from tqdm import tqdm
from six.moves import xrange
from urllib.request import urlretrieve
#from scipy.misc import imsave, imread, imresize
from tensorflow.examples.tutorials.mnist import input_data

# comment below two lines to implement this code with GPU
#os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(DIR_PATH, "data")  # path for your result

NUM_IMAGES = 60000
DATABASE_NAME = 'mnist'
OUTPUT_NAME = 'out_vanilla'
OUTPUT_PATH = os.path.join(DATA_DIR, OUTPUT_NAME)


# This section is to implement vanilla GAN. You need to complete TODO by yourself according to vanilla GAN theory.
class VanillaGAN():
    def __init__(self, sess, img_size, z_dim, hidden_dim, batch_size, epoch):
        self.sess = sess
        self.epoch = epoch
        self.z_dim = z_dim
        self.img_size = img_size
        self.img_dim = img_size * img_size
        self.img_shape = [img_size, img_size, 1]
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.build_model()
        self.model_name = "vanilla_GAN.model"

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.img = tf.placeholder(tf.float32, [None, self.img_dim], name='real_images')

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = tf.summary.histogram('z', self.z)

        initializer = tf.contrib.layers.xavier_initializer()

        # parameters for generator
        # input layer parameters ()
        self.G_W1 = tf.Variable(initializer([self.z_dim, self.hidden_dim]))
        self.G_b1 = tf.Variable(tf.zeros(shape=[self.hidden_dim]))

        # output layer parameters
        self.G_W2 = tf.Variable(initializer([self.hidden_dim, self.img_dim]))
        self.G_b2 = tf.Variable(tf.zeros(shape=[self.img_dim]))

        # parameters for discriminator
        # input layer parameters
        self.D_W1 = tf.Variable(initializer([self.img_dim, self.hidden_dim]))
        self.D_b1 = tf.Variable(tf.zeros(shape=[self.hidden_dim]))

        # output layer parameters
        self.D_W2 = tf.Variable(initializer([self.hidden_dim, 1]))
        self.D_b2 = tf.Variable(tf.zeros(shape=[1]))

        # self.img_fake is produced by generator with a random input z
        self.img_fake = self.generator(self.z, self.G_W1, self.G_b1, self.G_W2, self.G_b2)

        # self.D presents
        # self.D_fake presents
        self.D, self.D_logits_real = self.discriminator(self.img, self.D_W1, self.D_b1, self.D_W2, self.D_b2)
        self.D_fake, self.D_logits_fake = self.discriminator(self.img_fake, self.D_W1, self.D_b1, self.D_W2, self.D_b2)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_fake)
        self.img_fake_sum = tf.summary.histogram("G", self.img_fake)

        self.D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_real, labels=tf.ones_like(self.D_logits_real)))
        self.D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_fake,
                                                    labels=tf.zeros_like(self.D_logits_fake)))
        self.D_loss = self.D_loss_real + self.D_loss_fake
        self.G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_fake, labels=tf.ones_like(self.D_logits_fake)))

        self.D_vars = [self.D_W1, self.D_b1, self.D_W2, self.D_b2]
        self.G_vars = [self.G_W1, self.G_b1, self.G_W2, self.G_b2]

    def generator(self, z, W1, b1, W2, b2):
        # TODO: generate fake image with W1, b1, W2, b2
        # Hints: using tf.nn.relu(), tf.matmul(), tf.nn.sigmoid() to implement functions as:

        # The first function is: h1 = relu(z*W1 + b1),  
        h1 = tf.nn.relu(tf.matmul(z, W1) + b1)

        # The seconde one is: prob = h1*W2 + b2
        prob = tf.matmul(h1, W2) + b2

        # The last one is to use activation fuction sigmoid(prob)  
        output = tf.nn.sigmoid(prob)

        return output

    def discriminator(self, img, W1, b1, W2, b2):
        # TODO: discriminator is used to discriminate whether an input image is real or not.

        # The first function is: h1 = relu(img*W1 + b1)
        h1 = tf.nn.relu(tf.matmul(img, W1) + b1)

        # The second one is: logit = h1*W2 + b2
        logit = tf.matmul(h1, W2) + b2

        # The third one is to generate the probability of an input image to be a real one with sigmoid.
        prob = tf.nn.sigmoid(logit)

        return prob, logit

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
        d_optim = tf.train.AdamOptimizer().minimize(self.D_loss, var_list=self.D_vars)
        g_optim = tf.train.AdamOptimizer().minimize(self.G_loss, var_list=self.G_vars)

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

            # Update D network
            _, D_loss_curr = self.sess.run([d_optim, self.D_loss], feed_dict={self.img: input_imgs, self.z: input_z})
            _, G_loss_curr = self.sess.run([g_optim, self.G_loss], feed_dict={self.z: input_z})

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
    flags = tf.app.flags
    flags.DEFINE_integer("img_size", 28, "Image size.")
    flags.DEFINE_integer("z_dim", 100, "The dimension of random input z.")
    flags.DEFINE_integer("hidden_dim", 128, "The dimension of hidden ")
    flags.DEFINE_integer("batch_size", 16, "The size of batch images [32]")
    flags.DEFINE_integer("epoch", 100000, "Epoch to train!")
    FLAGS = flags.FLAGS

    config = tf.ConfigProto(
        device_count={'GPU': 1})  # if you wanna implement this code with GPU, change 0 to 1 or 2.
    with tf.Session(config=config) as sess:
        vanilla_gan = VanillaGAN(sess, img_size=FLAGS.img_size, z_dim=FLAGS.z_dim,
                                 hidden_dim=FLAGS.hidden_dim, batch_size=FLAGS.batch_size,
                                 epoch=FLAGS.epoch)
        vanilla_gan.train(OUTPUT_PATH)


if __name__ == "__main__":
    run()
