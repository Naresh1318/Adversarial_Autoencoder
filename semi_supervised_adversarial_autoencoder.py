import tensorflow as tf
import numpy as np
import datetime
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tensorflow.examples.tutorials.mnist import input_data

# Get the MNIST data
mnist = input_data.read_data_sets('./Data', one_hot=True)

# Parameters
input_dim = 784
n_l1 = 1000
n_l2 = 1000
z_dim = 10
batch_size = 100
n_epochs = 1000
learning_rate = 0.001
beta1 = 0.9
results_path = './Results/Semi_Supervised'
n_labels = 10
n_labeled = 1000

# Placeholders for input data and the targets
x_input = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim], name='Input')
x_input_l = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim], name='Labeled_Input')
y_input = tf.placeholder(dtype=tf.float32, shape=[batch_size, n_labels], name='Labels')
x_target = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim], name='Target')
real_distribution = tf.placeholder(dtype=tf.float32, shape=[batch_size, z_dim], name='Real_distribution')
categorial_distribution = tf.placeholder(dtype=tf.float32, shape=[batch_size, n_labels],
                                         name='Categorical_distribution')
manual_decoder_input = tf.placeholder(dtype=tf.float32, shape=[1, z_dim + n_labels], name='Decoder_input')


def form_results():
    """
    Forms folders for each run to store the tensorboard files, saved models and the log files.
    :return: three string pointing to tensorboard, saved models and log paths respectively.
    """
    folder_name = "/{0}_{1}_{2}_{3}_{4}_{5}_Semi_Supervised". \
        format(datetime.datetime.now(), z_dim, learning_rate, batch_size, n_epochs, beta1)
    tensorboard_path = results_path + folder_name + '/Tensorboard'
    saved_model_path = results_path + folder_name + '/Saved_models/'
    log_path = results_path + folder_name + '/log'
    if not os.path.exists(results_path + folder_name):
        os.mkdir(results_path + folder_name)
        os.mkdir(tensorboard_path)
        os.mkdir(saved_model_path)
        os.mkdir(log_path)
    return tensorboard_path, saved_model_path, log_path


def generate_image_grid(sess, op):
    """
    Generates a grid of images by passing a set of numbers to the decoder and getting its output.
    :param sess: Tensorflow Session required to get the decoder output
    :param op: Operation that needs to be called inorder to get the decoder output
    :return: None, displays a matplotlib window with all the merged images.
    """
    nx, ny = 10, 10
    random_inputs = np.random.randn(10, z_dim) * 5.
    sample_y = np.identity(10)
    plt.subplot()
    gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)
    i = 0
    for r in random_inputs:
        for t in sample_y:
            r, t = np.reshape(r, (1, z_dim)), np.reshape(t, (1, n_labels))
            dec_input = np.concatenate((t, r), 1)
            x = sess.run(op, feed_dict={manual_decoder_input: dec_input})
            ax = plt.subplot(gs[i])
            i += 1
            img = np.array(x.tolist()).reshape(28, 28)
            ax.imshow(img, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('auto')
    plt.show()


def dense(x, n1, n2, name):
    """
    Used to create a dense layer.
    :param x: input tensor to the dense layer
    :param n1: no. of input neurons
    :param n2: no. of output neurons
    :param name: name of the entire dense layer.i.e, variable scope name.
    :return: tensor with shape [batch_size, n2]
    """
    with tf.variable_scope(name, reuse=None):
        weights = tf.get_variable("weights", shape=[n1, n2],
                                  initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
        bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
        out = tf.add(tf.matmul(x, weights), bias, name='matmul')
        return out


# The autoencoder network
def encoder(x, reuse=False, supervised=False):
    """
    Encode part of the autoencoder.
    :param x: input to the autoencoder
    :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
    :param supervised: True -> returns output without passing it through softmax,
                       False -> returns output after passing it through softmax.
    :return: tensor which is the classification output and a hidden latent variable of the autoencoder.
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Encoder'):
        e_dense_1 = tf.nn.relu(dense(x, input_dim, n_l1, 'e_dense_1'))
        e_dense_2 = tf.nn.relu(dense(e_dense_1, n_l1, n_l2, 'e_dense_2'))
        latent_variable = dense(e_dense_2, n_l2, z_dim, 'e_latent_variable')
        cat_op = dense(e_dense_2, n_l2, n_labels, 'e_label')
        if not supervised:
            softmax_label = tf.nn.softmax(logits=cat_op, name='e_softmax_label')
        else:
            softmax_label = cat_op
        return softmax_label, latent_variable


def decoder(x, reuse=False):
    """
    Decoder part of the autoencoder.
    :param x: input to the decoder
    :param reuse: True -> Reuse the decoder variables, False -> Create or search of variables before creating
    :return: tensor which should ideally be the input given to the encoder.
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Decoder'):
        d_dense_1 = tf.nn.relu(dense(x, z_dim + n_labels, n_l2, 'd_dense_1'))
        d_dense_2 = tf.nn.relu(dense(d_dense_1, n_l2, n_l1, 'd_dense_2'))
        output = tf.nn.sigmoid(dense(d_dense_2, n_l1, input_dim, 'd_output'))
        return output


def discriminator_gauss(x, reuse=False):
    """
    Discriminator that is used to match the posterior distribution with a given gaussian distribution.
    :param x: tensor of shape [batch_size, z_dim]
    :param reuse: True -> Reuse the discriminator variables,
                  False -> Create or search of variables before creating
    :return: tensor of shape [batch_size, 1]
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Discriminator_Gauss'):
        dc_den1 = tf.nn.relu(dense(x, z_dim, n_l1, name='dc_g_den1'))
        dc_den2 = tf.nn.relu(dense(dc_den1, n_l1, n_l2, name='dc_g_den2'))
        output = dense(dc_den2, n_l2, 1, name='dc_g_output')
        return output


def discriminator_categorical(x, reuse=False):
    """
    Discriminator that is used to match the posterior distribution with a given categorical distribution.
    :param x: tensor of shape [batch_size, n_labels]
    :param reuse: True -> Reuse the discriminator variables,
                  False -> Create or search of variables before creating
    :return: tensor of shape [batch_size, 1]
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Discriminator_Categorial'):
        dc_den1 = tf.nn.relu(dense(x, n_labels, n_l1, name='dc_c_den1'))
        dc_den2 = tf.nn.relu(dense(dc_den1, n_l1, n_l2, name='dc_c_den2'))
        output = dense(dc_den2, n_l2, 1, name='dc_c_output')
        return output


def next_batch(x, y, batch_size):
    """
    Used to return a random batch from the given inputs.
    :param x: Input images of shape [None, 784]
    :param y: Input labels of shape [None, 10]
    :param batch_size: integer, batch size of images and labels to return
    :return: x -> [batch_size, 784], y-> [batch_size, 10]
    """
    index = np.arange(n_labeled)
    random_index = np.random.permutation(index)[:batch_size]
    return x[random_index], y[random_index]


def train(train_model=True):
    """
    Used to train the autoencoder by passing in the necessary inputs.
    :param train_model: True -> Train the model, False -> Load the latest trained model and show the image grid.
    :return: does not return anything
    """

    # Reconstruction Phase
    with tf.variable_scope(tf.get_variable_scope()):
        encoder_output_label, encoder_output_latent = encoder(x_input)
        # Concat class label and the encoder output
        decoder_input = tf.concat([encoder_output_label, encoder_output_latent], 1)
        decoder_output = decoder(decoder_input)

    # Regularization Phase
    with tf.variable_scope(tf.get_variable_scope()):
        d_g_real = discriminator_gauss(real_distribution)
        d_g_fake = discriminator_gauss(encoder_output_latent, reuse=True)

    with tf.variable_scope(tf.get_variable_scope()):
        d_c_real = discriminator_categorical(categorial_distribution)
        d_c_fake = discriminator_categorical(encoder_output_label, reuse=True)

    # Semi-Supervised Classification Phase
    with tf.variable_scope(tf.get_variable_scope()):
        encoder_output_label_, _ = encoder(x_input_l, reuse=True, supervised=True)

    # Generate output images
    with tf.variable_scope(tf.get_variable_scope()):
        decoder_image = decoder(manual_decoder_input, reuse=True)

    # Classification accuracy of encoder
    correct_pred = tf.equal(tf.argmax(encoder_output_label_, 1), tf.argmax(y_input, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Autoencoder loss
    autoencoder_loss = tf.reduce_mean(tf.square(x_target - decoder_output))

    # Gaussian Discriminator Loss
    dc_g_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_g_real), logits=d_g_real))
    dc_g_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_g_fake), logits=d_g_fake))
    dc_g_loss = dc_g_loss_fake + dc_g_loss_real

    # Categorical Discrimminator Loss
    dc_c_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_c_real), logits=d_c_real))
    dc_c_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_c_fake), logits=d_c_fake))
    dc_c_loss = dc_c_loss_fake + dc_c_loss_real

    # Generator loss
    generator_g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_g_fake), logits=d_g_fake))
    generator_c_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_c_fake), logits=d_c_fake))
    generator_loss = generator_c_loss + generator_g_loss

    # Supervised Encoder Loss
    supervised_encoder_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=encoder_output_label_))

    all_variables = tf.trainable_variables()
    dc_g_var = [var for var in all_variables if 'dc_g_' in var.name]
    dc_c_var = [var for var in all_variables if 'dc_c_' in var.name]
    en_var = [var for var in all_variables if 'e_' in var.name]

    # Optimizers
    autoencoder_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                   beta1=beta1).minimize(autoencoder_loss)
    discriminator_g_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                       beta1=beta1).minimize(dc_g_loss, var_list=dc_g_var)
    discriminator_c_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                       beta1=beta1).minimize(dc_c_loss, var_list=dc_c_var)
    generator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                 beta1=beta1).minimize(generator_loss, var_list=en_var)
    supervised_encoder_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                          beta1=beta1).minimize(supervised_encoder_loss,
                                                                                var_list=en_var)

    init = tf.global_variables_initializer()

    # Reshape immages to display them
    input_images = tf.reshape(x_input, [-1, 28, 28, 1])
    generated_images = tf.reshape(decoder_output, [-1, 28, 28, 1])

    # Tensorboard visualization
    tf.summary.scalar(name='Autoencoder Loss', tensor=autoencoder_loss)
    tf.summary.scalar(name='Discriminator gauss Loss', tensor=dc_g_loss)
    tf.summary.scalar(name='Discriminator categorical Loss', tensor=dc_c_loss)
    tf.summary.scalar(name='Generator Loss', tensor=generator_loss)
    tf.summary.scalar(name='Supervised Encoder Loss', tensor=supervised_encoder_loss)
    tf.summary.histogram(name='Encoder Gauss Distribution', values=encoder_output_latent)
    tf.summary.histogram(name='Real Gauss Distribution', values=real_distribution)
    tf.summary.histogram(name='Encoder Categorical Distribution', values=encoder_output_label)
    tf.summary.histogram(name='Real Categorical Distribution', values=categorial_distribution)
    tf.summary.image(name='Input Images', tensor=input_images, max_outputs=10)
    tf.summary.image(name='Generated Images', tensor=generated_images, max_outputs=10)
    summary_op = tf.summary.merge_all()

    # Saving the model
    saver = tf.train.Saver()
    step = 0
    with tf.Session() as sess:
        if train_model:
            tensorboard_path, saved_model_path, log_path = form_results()
            sess.run(init)
            writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=sess.graph)
            x_l, y_l = mnist.test.next_batch(n_labeled)
            for i in range(n_epochs):
                n_batches = int(n_labeled / batch_size)
                print("------------------Epoch {}/{}------------------".format(i, n_epochs))
                for b in range(1, n_batches + 1):
                    z_real_dist = np.random.randn(batch_size, z_dim) * 5.
                    real_cat_dist = np.random.randint(low=0, high=10, size=batch_size)
                    real_cat_dist = np.eye(n_labels)[real_cat_dist]
                    batch_x_ul, _ = mnist.train.next_batch(batch_size)
                    batch_x_l, batch_y_l = next_batch(x_l, y_l, batch_size=batch_size)
                    sess.run(autoencoder_optimizer, feed_dict={x_input: batch_x_ul, x_target: batch_x_ul})
                    sess.run(discriminator_g_optimizer,
                             feed_dict={x_input: batch_x_ul, x_target: batch_x_ul, real_distribution: z_real_dist})
                    sess.run(discriminator_c_optimizer,
                             feed_dict={x_input: batch_x_ul, x_target: batch_x_ul,
                                        categorial_distribution: real_cat_dist})
                    sess.run(generator_optimizer, feed_dict={x_input: batch_x_ul, x_target: batch_x_ul})
                    sess.run(supervised_encoder_optimizer, feed_dict={x_input_l: batch_x_l, y_input: batch_y_l})
                    if b % 5 == 0:
                        a_loss, d_g_loss, d_c_loss, g_loss, s_loss, summary = sess.run(
                            [autoencoder_loss, dc_g_loss, dc_c_loss, generator_loss, supervised_encoder_loss,
                             summary_op],
                            feed_dict={x_input: batch_x_ul, x_target: batch_x_ul,
                                       real_distribution: z_real_dist, y_input: batch_y_l, x_input_l: batch_x_l,
                                       categorial_distribution: real_cat_dist})
                        writer.add_summary(summary, global_step=step)
                        print("Epoch: {}, iteration: {}".format(i, b))
                        print("Autoencoder Loss: {}".format(a_loss))
                        print("Discriminator Gauss Loss: {}".format(d_g_loss))
                        print("Discriminator Categorical Loss: {}".format(d_c_loss))
                        print("Generator Loss: {}".format(g_loss))
                        print("Supervised Loss: {}\n".format(s_loss))
                        with open(log_path + '/log.txt', 'a') as log:
                            log.write("Epoch: {}, iteration: {}\n".format(i, b))
                            log.write("Autoencoder Loss: {}\n".format(a_loss))
                            log.write("Discriminator Gauss Loss: {}".format(d_g_loss))
                            log.write("Discriminator Categorical Loss: {}".format(d_c_loss))
                            log.write("Generator Loss: {}\n".format(g_loss))
                            log.write("Supervised Loss: {}".format(s_loss))
                    step += 1
                acc = 0
                num_batches = int(mnist.validation.num_examples/batch_size)
                for j in range(num_batches):
                    # Classify unseen validation data instead of test data or train data
                    batch_x_l, batch_y_l = mnist.validation.next_batch(batch_size=batch_size)
                    encoder_acc = sess.run(accuracy, feed_dict={x_input_l: batch_x_l, y_input: batch_y_l})
                    acc += encoder_acc
                acc /= num_batches
                print("Encoder Classification Accuracy: {}".format(acc))
                with open(log_path + '/log.txt', 'a') as log:
                    log.write("Encoder Classification Accuracy: {}".format(acc))
                saver.save(sess, save_path=saved_model_path, global_step=step)
        else:
            # Get the latest results folder
            all_results = os.listdir(results_path)
            all_results.sort()
            saver.restore(sess, save_path=tf.train.latest_checkpoint(results_path + '/' +
                                                                     all_results[-1] + '/Saved_models/'))
            generate_image_grid(sess, op=decoder_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Autoencoder Train Parameter")
    parser.add_argument('--train', '-t', type=bool, default=True,
                        help='Set to True to train a new model, False to load weights and display image grid')
    args = parser.parse_args()
    train(train_model=args.train)
