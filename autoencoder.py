import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Get the MNIST data
mnist = input_data.read_data_sets('./Data', one_hot=True)

# Parameters
input_dim = 784
n_l1 = 512
n_l2 = 256
n_l3 = 124
z_dim = 64
batch_size = 32
n_epochs = 100
learning_rate = 0.01

# Placeholders for input data and the targets
x_input = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim], name='Input')
x_target = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim], name='Target')


def dense(x, n1, n2, activation='relu'):
    weights = tf.Variable(tf.truncated_normal([n1, n2]), dtype=tf.float32)
    bias = tf.Variable(tf.truncated_normal([n2]), dtype=tf.float32)

    if activation == 'relu':
        out = tf.nn.relu(tf.add(tf.matmul(x, weights), bias))
    elif activation == 'sigmoid':
        out = tf.nn.sigmoid(tf.add(tf.matmul(x, weights), bias))
    elif activation is None:
        out = tf.add(tf.matmul(x, weights), bias)
    return out


# The autoencoder network
def autoencoder(x):
    # Encoder
    e_dense_1 = dense(x, input_dim, n_l1, activation='sigmoid')
    e_dense_2 = dense(e_dense_1, n_l1, n_l2, activation='sigmoid')
    e_dense_3 = dense(e_dense_2, n_l2, n_l3, activation='sigmoid')
    latent_variable = dense(e_dense_3, n_l3, z_dim, activation='sigmoid')

    # Decoder
    d_dense_1 = dense(latent_variable, z_dim, n_l3, activation='sigmoid')
    d_dense_2 = dense(d_dense_1, n_l3, n_l2, activation='sigmoid')
    d_dense_2 = dense(d_dense_2, n_l2, n_l1, activation='sigmoid')
    output = dense(d_dense_2, n_l1, input_dim, activation='sigmoid')
    return output


decoder_output = autoencoder(x_input)

# Loss
loss = tf.reduce_mean(tf.square(x_target - decoder_output))
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=decoder_output, labels=x_target))
# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
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
    writer = tf.summary.FileWriter(logdir='./Results/Tensorboard', graph=sess.graph)
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
            step += 1

        saver.save(sess, save_path='./Results/Saved_Models/', global_step=step)
