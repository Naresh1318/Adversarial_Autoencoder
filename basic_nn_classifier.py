import tensorflow as tf
import numpy as np
import os
import datetime
from tensorflow.examples.tutorials.mnist import input_data

# Parameters
input_dim = 784
n_l1 = 1000
n_l2 = 1000
batch_size = 100
n_epochs = 1000
learning_rate = 0.001
beta1 = 0.9
z_dim = 'NA'
results_path = './Results/Basic_NN_Classifier'
n_labels = 10
n_labeled = 1000

# Get MNIST data
mnist = input_data.read_data_sets('./Data', one_hot=True)

# Placeholders
x_input = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y_target = tf.placeholder(dtype=tf.float32, shape=[None, 10])


def form_results():
    """
    Forms folders for each run to store the tensorboard files, saved models and the log files.
    :return: three string pointing to tensorboard, saved models and log paths respectively.
    """
    folder_name = "/{0}_{1}_{2}_{3}_{4}_{5}_Basic_NN_Classifier". \
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


def dense(x, n1, n2, name):
    """
    Used to create a dense layer.
    :param x: input tensor to the dense layer
    :param n1: no. of input neurons
    :param n2: no. of output neurons
    :param name: name of the entire dense layer.
    :return: tensor with shape [batch_size, n2]
    """
    with tf.name_scope(name):
        weights = tf.Variable(tf.random_normal(shape=[n1, n2], mean=0., stddev=0.01), name='weights')
        bias = tf.Variable(tf.zeros(shape=[n2]), name='bias')
        output = tf.add(tf.matmul(x, weights), bias, name='output')
        return output


# Dense Network
def dense_nn(x):
    """
    Network used to classify MNIST digits.
    :param x: tensor with shape [batch_size, 784], input to the dense fully connected layer.
    :return: [batch_size, 10], logits of dense fully connected.
    """
    dense_1 = tf.nn.dropout(tf.nn.relu(dense(x, input_dim, n_l1, 'dense_1')), keep_prob=0.25)
    dense_2 = tf.nn.dropout(tf.nn.relu(dense(dense_1, n_l1, n_l2, 'dense_2')), keep_prob=0.25)
    dense_3 = dense(dense_2, n_l2, n_labels, 'dense_3')
    return dense_3


def train():
    """
    Used to train the autoencoder by passing in the necessary inputs.
    :return: does not return anything
    """
    dense_output = dense_nn(x_input)

    # Loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_output, labels=y_target))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(loss)

    # Accuracy
    pred_op = tf.equal(tf.argmax(dense_output, 1), tf.argmax(y_target, 1))
    accuracy = tf.reduce_mean(tf.cast(pred_op, dtype=tf.float32))

    # Summary
    tf.summary.scalar(name='Loss', tensor=loss)
    tf.summary.scalar(name='Accuracy', tensor=accuracy)
    summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()

    init = tf.global_variables_initializer()

    step = 0
    with tf.Session() as sess:
        tensorboard_path, saved_model_path, log_path = form_results()
        x_l, y_l = mnist.test.next_batch(n_labeled)
        writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=sess.graph)
        sess.run(init)
        for e in range(1, n_epochs + 1):
            n_batches = int(n_labeled / batch_size)
            for b in range(1, n_batches + 1):
                batch_x_l, batch_y_l = next_batch(x_l, y_l, batch_size=batch_size)
                sess.run(optimizer, feed_dict={x_input: batch_x_l, y_target: batch_y_l})
                if b % 5 == 0:
                    loss_, summary = sess.run([loss, summary_op], feed_dict={x_input: batch_x_l, y_target: batch_y_l})
                    writer.add_summary(summary, step)
                    print("Epoch: {} Iteration: {}".format(e, b))
                    print("Loss: {}".format(loss_))
                    with open(log_path + '/log.txt', 'a') as log:
                        log.write("Epoch: {}, iteration: {}\n".format(e, b))
                        log.write("Loss: {}\n".format(loss_))
                step += 1
            acc = 0
            num_batches = int(mnist.validation.num_examples / batch_size)
            for j in range(num_batches):
                # Classify unseen validation data instead of test data or train data
                batch_x_l, batch_y_l = mnist.validation.next_batch(batch_size=batch_size)
                val_acc = sess.run(accuracy, feed_dict={x_input: batch_x_l, y_target: batch_y_l})
                acc += val_acc
            acc /= num_batches
            print("Classification Accuracy: {}".format(acc))
            with open(log_path + '/log.txt', 'a') as log:
                log.write("Classification Accuracy: {}".format(acc))
            saver.save(sess, save_path=saved_model_path, global_step=step)

if __name__ == '__main__':
    train()
