import tensorflow as tf
import datetime
import os
from tensorflow.examples.tutorials.mnist import input_data

# Get the MNIST data
mnist = input_data.read_data_sets('./Data', one_hot=True)

# Parameters
z_dim = 2
learning_rate = 0.01
batch_size = 32
n_epochs = 100
beta1 = 0.5
results_path = './Results'

# Inputs and target
x_input = tf.placeholder(tf.float32, [None, 784], name='Input_Image')
x_target = tf.placeholder(tf.float32, [None, 784], name='Target_Image')
x_input_ = tf.reshape(x_input, [-1, 28, 28, 1])
x_target_ = tf.reshape(x_target, [-1, 28, 28, 1])


def form_results():
    folder_name = "/{0}_{1}_{2}_{3}_{4}_{5}". \
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


def cnn_2d(x, shape, name):
    with tf.name_scope(name):
        weights = tf.Variable(tf.truncated_normal(shape))
        bias = tf.Variable(tf.zeros([shape[-1]]))
        conv = tf.nn.conv2d(x, weights, [1, 2, 2, 1], padding='SAME') + bias
        return conv


def cnn_2d_trans(x, shape, output_shape, name):
    with tf.name_scope(name):
        weights = tf.Variable(tf.truncated_normal(shape))
        bias = tf.Variable(tf.zeros([shape[-2]]))
        conv_t = tf.nn.conv2d_transpose(x, weights, output_shape, strides=[1, 2, 2, 1], padding='SAME') + bias
        return conv_t


def dense(x, n1, n2, name, activation='relu'):
    with tf.name_scope(name):
        weights = tf.Variable(tf.truncated_normal([n1, n2]), dtype=tf.float32)
        bias = tf.Variable(tf.truncated_normal([n2]), dtype=tf.float32)

        if activation == 'relu':
            out = tf.nn.relu(tf.add(tf.matmul(x, weights), bias))
        elif activation == 'sigmoid':
            out = tf.nn.sigmoid(tf.add(tf.matmul(x, weights), bias))
        elif activation == 'tanh':
            out = tf.nn.tanh(tf.add(tf.matmul(x, weights), bias))
        elif activation is None:
            out = tf.add(tf.matmul(x, weights), bias)
        return out


def encoder(x):
    e_conv1 = tf.nn.tanh(cnn_2d(x, [3, 3, 1, 64], name='e_conv1'))
    e_conv2 = tf.nn.tanh(cnn_2d(e_conv1, [3, 3, 64, 32], name='e_conv2'))
    e_conv2_flat = tf.reshape(e_conv2, [-1, 7 * 7 * 32], name='Reshape_1')
    output = dense(e_conv2_flat, 7 * 7 * 32, z_dim, activation=None, name='den1')
    return output


def decoder(x):
    d_den1 = dense(x, z_dim, 7 * 7 * 32, activation='tanh', name='d_den1')
    d_den1_unflat = tf.reshape(d_den1, [-1, 7, 7, 32], name='Reshape_2')
    d_conv1 = tf.nn.tanh(cnn_2d_trans(d_den1_unflat, shape=[3, 3, 64, 32],
                                      output_shape=[batch_size, 14, 14, 64], name='d_conv1'))
    output = tf.nn.sigmoid(cnn_2d_trans(d_conv1, shape=[3, 3, 1, 64],
                                        output_shape=[batch_size, 28, 28, 1], name='d_conv2'))
    return output


def train(train_flag=True):
    encoder_output = encoder(x_input_)
    decoder_output = decoder(encoder_output)

    # Loss
    loss = tf.reduce_mean(tf.square(x_target_ - decoder_output))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(loss)
    init = tf.global_variables_initializer()

    # Visualization
    tf.summary.scalar(name='Loss', tensor=loss)
    tf.summary.image(name='Input Images', tensor=x_input_, max_outputs=10)
    tf.summary.image(name='Generated Images', tensor=decoder_output, max_outputs=10)
    summary_op = tf.summary.merge_all()

    # Saving the model
    saver = tf.train.Saver()
    step = 0
    with tf.Session() as sess:
        if train_flag:
            tensorboard_path, saved_model_path, log_path = form_results()
            sess.run(init)
            writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=sess.graph)
            for i in range(n_epochs):
                n_batches = int(mnist.train.num_examples / batch_size)
                for b in range(n_batches):
                    batch_x, _ = mnist.train.next_batch(batch_size)
                    sess.run(optimizer, feed_dict={x_input: batch_x, x_target: batch_x})
                    if b % 50 == 0:
                        batch_loss, summary = sess.run([loss, summary_op], feed_dict={x_input: batch_x, x_target: batch_x})
                        writer.add_summary(summary, global_step=step)
                        print("Loss: {}".format(batch_loss))
                        print("Epoch: {}, iteration: {}".format(i, b))
                        with open(log_path + '/log.txt', 'a') as log:
                            log.write("Epoch: {}, iteration: {}\n".format(i, b))
                            log.write("Loss: {}\n".format(batch_loss))
                    step += 1

                saver.save(sess, save_path=saved_model_path, global_step=step)
        else:
            # Get the latest results folder
            all_results = os.listdir(results_path)
            all_results.sort()
            saver.restore(sess, tf.train.latest_checkpoint(results_path + all_results[-1] + '/Saved_models/'))
            print("Model Loaded!")
            # TODO: give any desired input to the encoder or decoder

if __name__ == '__main__':
    train(True)
