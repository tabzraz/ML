import numpy as np
import tensorflow as tf


class BayesianFC:

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Variational Parameters
        self.qw_mean = tf.Variable(initial_value=np.zeros(shape=(input_dim, output_dim)), dtype=tf.float32)
        self.qw_p = tf.Variable(initial_value=np.zeros(shape=(input_dim, output_dim)), dtype=tf.float32)
        self.qb_mean = tf.Variable(initial_value=np.ones(shape=(output_dim)) * 0.1, dtype=tf.float32)
        self.qb_p = tf.Variable(initial_value=np.zeros(shape=(output_dim)), dtype=tf.float32)

        # Prior Parameters
        # self.pw_mean = tf.constant(value=np.zeros(shape=(input_dim, output_dim)), dtype=tf.float32)
        # self.pw_sigma = tf.constant(value=np.ones(shape=(input_dim, output_dim)) * 1, dtype=tf.float32)
        # self.pb_mean = tf.constant(initial_value=np.ones(shape=(output_dim)) * 0.1, dtype=tf.float32)
        # self.pb_sigma = tf.constant(initial_value=np.ones(shape=(output_dim)) * 1, dtype=tf.float32)
        self.pw_mean = 0
        self.pw_sigma = 1
        self.pb_mean = 0.1
        self.pb_sigma = 1

        # Likelihood std
        # todo: change this to be an output of the network?
        self.likelihood_std = 1

    def output(self, input, W=None, b=None, sample=False):
        # If we have not provided the weights, then use our distribution to generate them
        if W is None:
            if sample:
                epsilon_w = tf.random_normal(shape=(self.input_dim, self.output_dim))
                epsilon_b = tf.random_normal(shape=(self.output_dim))
            else:
                epsilon_w = epsilon_b = 0

            W = self.qw_mean + tf.nn.softplus(self.qw_p) * epsilon_w
            b = self.qb_mean + tf.nn.softplus(self.qb_p) * epsilon_b

        # Now we have the weights, calculate the output of the network
        return tf.nn.relu(tf.matmul(input, W) + b)

    def gaussian_pdf(self, x, mu, sigma):
        return tf.exp(-((x - mu)**2)/2*(sigma**2))/tf.sqrt(2*np.pi*(sigma**2))

    def update(self, data, N):

        loss = 0

        for _ in range(N):
            # Sample weights
            epsilon_w = tf.random_normal(shape=(self.input_dim, self.output_dim))
            epsilon_b = tf.random_normal(shape=(self.output_dim))

            W_sigmas = tf.nn.softplus(self.qw_p)
            b_sigmas = tf.nn.softplus(self.qb_p)
            W = self.qw_mean + W_sigmas * epsilon_w
            b = self.qb_mean + b_sigmas * epsilon_b

            qw_i = tf.reduce_sum(self.gaussian_pdf(W_sigmas * epsilon_w, 0, W_sigmas))
            qw_i += tf.reduce_sum(self.gaussian_pdf(b_sigmas * epsilon_b, 0, b_sigmas))
            log_qw_i = tf.log(qw_i)

            pw_i = tf.reduce_sum(self.gaussian_pdf(W, self.pw_mean, self.pw_sigma))
            pw_i += tf.reduce_sum(self.gaussian_pdf(b, self.pb_mean, self.pb_sigma))
            log_pw_i = tf.log(pw_i)

            data_likelihood = 0
            for (x, y) in data:
                # We assume the nn is a probabilistic model P(y|x,w)
                output = self.output(x, W=W, b=b)
                data_likelihood += tf.reduce_sum(self.gaussian_pdf(y, output, self.likelihood_std))

            log_data_likelihood = tf.log(data_likelihood)

            loss += -log_qw_i - log_pw_i - log_data_likelihood








