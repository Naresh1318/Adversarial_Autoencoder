import tensorflow as tf
import datetime
import os
from tensorflow.examples.tutorials.mnist import input_data

# Get the MNIST data
mnist = input_data.read_data_sets('./Data', one_hot=True)

# Parameters
input_dim = 784
n_l1 = 1000
n_l2 = 1000
z_dim = 2
batch_size = 100
n_epochs = 1000
learning_rate = 0.001
beta1 = 0.9
results_path = './Results'


# Placeholders for input data and the targets
x_input = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim], name='Input')
x_target = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim], name='Target')


def form_results():
    folder_name = "/{0}_{1}_{2}_{3}_{4}_{5}_V2". \
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


def dense(x, n1, n2, name):
    with tf.name_scope(name):
        weights = tf.Variable(tf.random_normal([n1, n2], mean=0., stddev=0.01), dtype=tf.float32, name='weights')
        bias = tf.Variable(tf.constant(value=0., shape=[n2]), dtype=tf.float32, name='bias')
        out = tf.add(tf.matmul(x, weights), bias, name='matmul')
        return out


# The autoencoder network
def autoencoder(x):
    # Encoder
    e_dense_1 = tf.nn.relu(dense(x, input_dim, n_l1, 'e_dense_1'))
    e_dense_2 = tf.nn.relu(dense(e_dense_1, n_l1, n_l2, 'e_dense_2'))
    latent_variable = dense(e_dense_2, n_l2, z_dim, 'e_latent_variable')

    # Decoder
    d_dense_1 = tf.nn.relu(dense(latent_variable, z_dim, n_l2, 'd_dense_1'))
    d_dense_2 = tf.nn.relu(dense(d_dense_1, n_l2, n_l1, 'd_dense_2'))
    output = tf.nn.sigmoid(dense(d_dense_2, n_l1, input_dim, 'd_output'))
    return output


def train():
    decoder_output = autoencoder(x_input)

    # Loss
    loss = tf.reduce_mean(tf.square(x_target - decoder_output))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(loss)
    init = tf.global_variables_initializer()

    # Visualization
    tf.summary.scalar(name='Loss', tensor=loss)
    input_images = tf.reshape(x_input, [-1, 28, 28, 1])
    generated_images = tf.reshape(decoder_output, [-1, 28, 28, 1])
    tf.summary.image(name='Input Images', tensor=input_images, max_outputs=10)
    tf.summary.image(name='Generated Images', tensor=generated_images, max_outputs=10)
    summary_op = tf.summary.merge_all()

    # Saving the model
    saver = tf.train.Saver()
    step = 0
    with tf.Session() as sess:
        sess.run(init)
        tensorboard_path, saved_model_path, log_path = form_results()
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

if __name__ == '__main__':
    train()
