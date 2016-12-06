import numpy as np
import tensorflow as tf


class BayesianFC:

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Parameters
        self.qw_mean = tf.Variable(initial_value=np.zeros(shape=(input_dim, output_dim)), dtype=tf.float32)
        self.qw_p = tf.Variable(initial_value=np.zeros(shape=(input_dim, output_dim)), dtype=tf.float32)
        self.qb_mean = tf.Variable(initial_value=np.ones(shape=(output_dim)) * 0.1, dtype=tf.float32)
        self.qb_p = tf.Variable(initial_value=np.zeros(shape=(output_dim)), dtype=tf.float32)

    def output(self, input, sample=False):
        if not sample:
            epsilon_w = epsilon_b = 0
        else:
            epsilon_w = tf.random_normal(shape=(self.input_dim, self.output_dim))
            epsilon_b = tf.random_normal(shape=(self.output_dim))

        W = self.qw_mean + tf.nn.softplus(self.qw_p) * epsilon_w
        b = self.qb_mean + tf.nn.softplus(self.qb_p) * epsilon_b

        return tf.nn.relu(tf.matmul(input, W) + b)

    def update(self, data):



