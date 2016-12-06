import numpy as np
import tensorflow as tf


class BayesianFC:

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Parameters
        self.w_mean = tf.Variable(initial_value=np.zeros(shape=(input_dim, output_dim)), dtype=tf.float32)
        self.w_p = tf.Variable(initial_value=np.zeros(shape=(input_dim, output_dim)), dtype=tf.float32)
        self.b_mean = tf.Variable(initial_value=np.ones(shape=(output_dim)) * 0.1, dtype=tf.float32)
        self.b_p = tf.Variable(initial_value=np.zeros(shape=(output_dim)), dtype=tf.float32)

    def output(self, input, sample=False):
        if not sample:
            epsilon_w = epsilon_b = 0
        else:
            epsilon_w = tf.random_normal(shape=(self.input_dim, self.output_dim))
            epsilon_b = tf.random_normal(shape=(self.output_dim))

        W = self.w_mean + tf.nn.softplus(self.w_p) * epsilon_w
        b = self.b_mean + tf.nn.softplus(self.b_p) * epsilon_b

        return tf.nn.relu(tf.matmul(input, W) + b)


