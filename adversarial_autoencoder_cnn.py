import tensorflow as tf
import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

# Get data
mnist = input_data.read_data_sets('./Data', one_hot=True)

# Parameters
z_dim = 100
learning_rate = 0.001
batch_size = 32
n_epochs = 100
TINY = 1e-15

# Placeholders
x_input = tf.placeholder(tf.float32, [None, 784], name='input_image')
x_target = tf.placeholder(tf.float32, [None, 784], name='target')
x_input_ = tf.reshape(x_input, [-1, 28, 28, 1])
x_target_ = tf.reshape(x_target, [-1, 28, 28, 1])
real_dist_input = tf.placeholder(tf.float32, [None, 100], name='real_distribution')


# TODO: Changed bias initialization from zeros to truncated norm
def cnn_2d(x, shape, name):
    with tf.variable_scope(name, reuse=None):
        weights = tf.get_variable("weights", shape=shape,
                                  initializer=tf.truncated_normal_initializer(mean=0., stddev=1.))
        bias = tf.get_variable("bias", [shape[-1]], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(x, weights, [1, 2, 2, 1], padding='SAME') + bias
        return conv


def cnn_2d_trans(x, shape, output_shape, name):
    with tf.variable_scope(name, reuse=None):
        weights = tf.get_variable("weights", shape=shape,
                                  initializer=tf.truncated_normal_initializer(mean=0., stddev=1.))
        bias = tf.get_variable("bias", [shape[-2]], initializer=tf.constant_initializer(0.0))
        conv_t = tf.nn.conv2d_transpose(x, weights, output_shape, strides=[1, 2, 2, 1], padding='SAME') + bias
        return conv_t


def dense(x, n1, n2, name):
    with tf.variable_scope(name, reuse=None):
        weights = tf.get_variable("weights", shape=[n1, n2],
                                  initializer=tf.truncated_normal_initializer(mean=0., stddev=1.))
        bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
        out = tf.add(tf.matmul(x, weights), bias)
        return out


def encoder(x, reuse=False):
    # Encoder
    if reuse:
        tf.get_variable_scope().reuse_variables()
    e_conv1 = tf.nn.tanh(cnn_2d(x, [3, 3, 1, 64], name='e_conv1'))
    e_conv2 = tf.nn.tanh(cnn_2d(e_conv1, [3, 3, 64, 32], name='e_conv2'))
    e_conv2_flat = tf.reshape(e_conv2, [-1, 7 * 7 * 32], name='e_Reshape_1')
    den1 = dense(e_conv2_flat, 7 * 7 * 32, 100, name='e_den1')
    return den1


def decoder(x):
    # Decoder
    d_den1 = tf.nn.tanh(dense(x, 100, 7 * 7 * 32, name='d_den1'))
    d_den1_unflat = tf.reshape(d_den1, [-1, 7, 7, 32], name='d_Reshape_2')
    d_conv1 = tf.nn.tanh(cnn_2d_trans(d_den1_unflat, shape=[3, 3, 64, 32],
                                      output_shape=[batch_size, 14, 14, 64], name='d_conv1'))
    output = tf.nn.sigmoid(cnn_2d_trans(d_conv1, shape=[3, 3, 1, 64],
                                        output_shape=[batch_size, 28, 28, 1], name='d_conv2'))
    return output


def discriminator(x, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    dc_den1 = tf.nn.relu(dense(x, 100, 500, name='dc_den1'))
    dc_den2 = tf.nn.relu(dense(dc_den1, 500, 500, name='dc_den2'))
    output = dense(dc_den2, 500, 1, name='dc_output')
    return output

encoder_output = encoder(x_input_)
decoder_output = decoder(encoder_output)

# Reconstruction loss
recon_loss = tf.reduce_mean(tf.square(x_target_ - decoder_output))
# TODO: Try out cross-entropy cost
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=decoder_output, labels=x_target))

# Reconstruction Optimizer
recon_optimizer = tf.train.AdamOptimizer(learning_rate=0.0006).minimize(recon_loss)

# Fake input to discriminator
with tf.variable_scope(tf.get_variable_scope()) as enc_scope:
    fake_dist_input = encoder(x_input_, reuse=True)

# Compute discriminator outputs and loss
with tf.variable_scope(tf.get_variable_scope()) as dis_scope:
    d_real = discriminator(real_dist_input)
    d_fake = discriminator(fake_dist_input, reuse=True)
    # dc_loss = -tf.reduce_mean(tf.log(d_real + TINY) + tf.log(1 - d_fake + TINY))
    dc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(targets=tf.ones_like(d_real), logits=d_real))
    dc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(targets=tf.zeros_like(d_fake), logits=d_fake))
    dc_loss = dc_loss_fake + dc_loss_real

all_variables = tf.trainable_variables()
dc_var = [var for var in all_variables if 'dc_' in var.name]
en_var = [var for var in all_variables if 'e_' in var.name]

# Optimizer for Discriminator
dc_optimizer = tf.train.AdamOptimizer(learning_rate=0.0008).minimize(dc_loss, var_list=dc_var)

# Generator loss and optimizer
# gen_loss = -tf.reduce_mean(tf.log(d_fake + TINY))
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(targets=tf.ones_like(d_fake), logits=d_fake))
gen_optimizer = tf.train.AdamOptimizer(learning_rate=0.0008).minimize(gen_loss, var_list=en_var)

init = tf.global_variables_initializer()

# Visualization
# PLot of losses
tf.summary.scalar(name='Reconstruction Loss', tensor=recon_loss)
tf.summary.scalar(name='Discriminator Loss', tensor=dc_loss)
tf.summary.scalar(name='Generator Loss', tensor=gen_loss)

# Histogram of distributions obtained
tf.summary.histogram(name='Real_distribution', values=real_dist_input)
tf.summary.histogram(name='Encoder_distribution', values=fake_dist_input)

# Images given as inputs and output images
tf.summary.image(name='Input Images', tensor=x_input_, max_outputs=10)
tf.summary.image(name='Generated Images', tensor=decoder_output, max_outputs=10)
summary_op = tf.summary.merge_all()

# Saving the model
saver = tf.train.Saver()
step = 0
log_dir = './Results/Tensorboard'
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter(logdir=log_dir, graph=sess.graph)
    print("Trainig...")
    print("Tensorboard summaries saved in: {}".format(log_dir))
    for i in range(n_epochs):
        n_batches = int(mnist.train.num_examples / batch_size)
        t = 0.
        for b in range(1, n_batches + 1):
            t1 = time.time()
            z_real_dist = np.random.randn(batch_size, 100) * 5.
            batch_x, _ = mnist.train.next_batch(batch_size)
            sess.run(recon_optimizer, feed_dict={x_input: batch_x, x_target: batch_x})
            sess.run(dc_optimizer, feed_dict={x_input: batch_x, x_target: batch_x, real_dist_input: z_real_dist})
            sess.run(gen_optimizer, feed_dict={x_input: batch_x, x_target: batch_x})
            t += (time.time() - t1)
            if b % 50 == 0:
                reconstruction_loss, discriminator_loss, generator_loss, summary = sess.run(
                    [recon_loss, dc_loss, gen_loss, summary_op],
                    feed_dict={x_input: batch_x, x_target: batch_x, real_dist_input: z_real_dist})
                writer.add_summary(summary, global_step=step)
                print("Epoch: {}, iteration: {}".format(i, b))
                print("Reconstruction Loss: {}".format(reconstruction_loss))
                print("Discriminator Loss: {}".format(discriminator_loss))
                print("Generator Loss: {}".format(generator_loss))
                print("Average time taken for 50 batchs: {}".format(t/50))
                t = 0.
            step += 1
        saver.save(sess, save_path='./Results/Saved_Models/', global_step=step)

    # TODO: Restore model
