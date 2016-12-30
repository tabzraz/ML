import numpy as np
import tensorflow as tf


class Bayesian_FC:

    def __init__(self, input_dim, output_dim,
                 # Variational
                 qw_mean_initial=0.0,
                 qw_std_initial=-1.0,
                 qb_mean_initial=0.0,
                 qb_std_initial=-1.0,
                 # Prior
                 pw_mean=0.0,
                 pw_sigma=2.0,
                 pb_mean=0.0,
                 pb_sigma=2.0,
                 # Model
                 activation=tf.nn.relu,
                 name="Bayesian_FC_Layer"
                 ):

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Hyperparameters
        self.qw_mean_initial = qw_mean_initial
        self.qw_logstd_initial = np.log(np.exp(qw_std_initial) - 1)
        self.qb_mean_initial = qb_mean_initial
        self.qb_logstd_initial = np.log(np.exp(qb_std_initial) - 1)

        # Trainable Variables
        with tf.name_scope(name):
            with tf.name_scope("Weights"):
                qw_mean = tf.Variable(initial_value=np.ones(shape=(input_dim, output_dim)) * self.qw_mean_initial, dtype=tf.float32, name="Weight_means")
                qw_p = tf.Variable(initial_value=np.ones(shape=(input_dim, output_dim)) * self.qw_logstd_initial, dtype=tf.float32, name="Weight_lodstds")
            with tf.name_scope("Biases"):
                qb_mean = tf.Variable(initial_value=np.ones(shape=(output_dim,)) * self.qb_mean_initial, dtype=tf.float32, name="Bias_means")
                qb_p = tf.Variable(initial_value=np.ones(shape=(output_dim,)) * self.qb_logstd_initial, dtype=tf.float32, name="Bias_logstds")

        self.qw_mean = qw_mean
        self.qw_p = qw_p
        self.qw_sigma = tf.nn.softplus(qw_p)
        self.qb_mean = qb_mean
        self.qb_p = qb_p
        self.qb_sigma = tf.nn.softplus(qb_p)

        # Prior hyperparameters
        self.pw_mean = pw_mean
        self.pw_sigma = pw_sigma
        self.pb_mean = pb_mean
        self.pb_sigma = pb_sigma

        # Model
        self.activation = activation

    def sample(self, input_tensor, use_mean=False):
        epsilon_w = tf.random_normal(shape=(self.input_dim, self.output_dim))
        epsilon_b = tf.random_normal(shape=(self.output_dim, ))

        if use_mean:
            epsilon_w = 0.0
            epsilon_b = 0.0

        W = self.qw_mean + tf.nn.softplus(self.qw_p) * epsilon_w
        b = self.qb_mean + tf.nn.softplus(self.qb_p) * epsilon_b

        sample_output = self.activation(tf.matmul(input_tensor, W) + b)

        return sample_output

    def local_reparam_sample(self, input_tensor):
        x = input_tensor
        batch_size = tf.shape(x)[0]
        qw_sigma = tf.nn.softplus(self.qw_p)
        qb_sigma = tf.nn.softplus(self.qb_p)

        gamma = tf.matmul(x, self.qw_mean) + self.qb_mean
        # print(gamma)
        delta = tf.matmul(tf.square(x), tf.square(qw_sigma)) + tf.square(qb_sigma)
        # print(delta)
        epsilon = tf.random_normal(shape=(batch_size, self.output_dim))
        # print(epsilon)

        noisy_activations = gamma + tf.sqrt(delta) * epsilon

        return self.activation(noisy_activations)

    def kl(self):
        return kl_q_p(self.qw_mean, self.qw_sigma, self.pw_mean, self.pw_sigma) + kl_q_p(self.qb_mean, self.qb_sigma, self.pb_mean, self.pb_sigma)


def kl_q_p(q_mean, q_sigma, p_mean, p_sigma):
    log_term = tf.log(p_sigma / (q_sigma + 1e-3))
    mean_term = (tf.square(q_sigma) + tf.square(q_mean - p_mean)) / (2.0 * tf.square(p_sigma) + 1e-3)
    return tf.reduce_sum(log_term + mean_term) - tf.size(q_mean, out_type=tf.float32) * 0.5
