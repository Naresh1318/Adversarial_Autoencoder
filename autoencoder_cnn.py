import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Get the MNIST data
mnist = input_data.read_data_sets('./Data', one_hot=True)

# Parameters
