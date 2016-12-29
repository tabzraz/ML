import numpy as np
import tensorflow as tf


class Bayesian_Net:

    def __init__(self, likelihood_std=0.1):
        self.likelihood_std = likelihood_std
        self.priors = []
        self.variationals = []
        self.kls = []

        sampling = tf.placeholder(dtype=tf.bool, shape=[])
        self.sampling = sampling

    def fully_connected(self, input_tensor, output_dim,
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
                        name="BayesianFC",
                        activation=tf.nn.relu
                        ):

        input_shape = tf.shape(input_tensor[0])
        input_dim = tf.reduce_prod(input_shape)

        # Trainable Variables
        with tf.name_scope(name):
            with tf.name_scope("Weights"):
                qw_mean = tf.Variable(initial_value=np.ones(shape=(input_dim, output_dim)) * self.qw_mean_initial, dtype=tf.float32, name="Weight_means")
                qw_p = tf.Variable(initial_value=np.ones(shape=(input_dim, output_dim)) * self.qw_logstd_initial, dtype=tf.float32, name="Weight_lodstds")
            with tf.name_scope("Biases"):
                qb_mean = tf.Variable(initial_value=np.ones(shape=(output_dim,)) * self.qb_mean_initial, dtype=tf.float32, name="Bias_means")
                qb_p = tf.Variable(initial_value=np.ones(shape=(output_dim,)) * self.qb_logstd_initial, dtype=tf.float32, name="Bias_logstds")

        qw_sigma = tf.nn.softplus(qw_p)
        qb_sigma = tf.nn.softplus(qb_p)

        layer_q_vars = [[(qw_mean, qw_p), (qb_mean, qb_p)]]
        layer_prior_vars = [[(pw_mean, pw_sigma), (pb_mean, pb_sigma)]]

        self.variationals += layer_q_vars
        self.priors += layer_prior_vars

        def layer_kls():
            return kl_q_p(qw_mean, qw_sigma, pw_mean, pw_sigma) + kl_q_p(qb_mean, qb_sigma, pb_mean, pb_sigma)

        self.kls += layer_kls

        batch_size = tf.shape(input_tensor)[0]

        # Local reparametrisation trick
        gamma = tf.matmul(input_tensor, qw_mean) + qb_mean
        # print(gamma)
        delta = tf.matmul(tf.square(input_tensor), tf.square(qw_sigma)) + tf.square(qb_sigma)
        # print(delta)
        epsilon = tf.random_normal(shape=(batch_size, output_dim))
        # print(epsilon)

        noisy_activations = gamma + tf.sqrt(delta) * epsilon

        local_reparam_trick_output = activation(noisy_activations)

        # Normal sampling (same weights for each element in batch)
        epsilon_w = tf.random_normal(shape=(input_dim, output_dim))
        epsilon_b = tf.random_normal(shape=(output_dim, ))

        W = qw_mean + qw_sigma * epsilon_w
        b = qb_mean + qb_sigma * epsilon_b

        sampling_output = activation(tf.matmul(input_tensor, W) + b)

        return tf.cond(self.sampling, sampling_output, local_reparam_trick_output)



    def loss(self, N=4):
        data_input = tf.placeholder(tf.float32, shape=[None, self.layer_dims[0]])
        data_target = tf.placeholder(tf.float32, shape=[None, self.layer_dims[-1]])
        minibatch_kl_scaling = tf.placeholder(tf.float32, shape=[])
        batch_size = tf.placeholder(tf.int32, shape=[])

        data_loss = 0.0
        for _ in range(N):
            prediction = self.sampled_output(data_input, batch_size)
            data_loss += tf.reduce_sum(self.log_gaussian_pdf(data_target, prediction, self.likelihood_std))

        kl_loss = 0.0
        for qw_mean, qw_p, qb_mean, qb_p, input_dim, output_dim in\
                zip(self.qw_means, self.qw_ps, self.qb_means, self.qb_ps, self.layer_dims[:-1], self.layer_dims[1:]):
            qw_sigma = tf.nn.softplus(qw_p)
            qb_sigma = tf.nn.softplus(qb_p)
            kl_loss += self.kl_q_p(qw_mean, qw_sigma, self.pw_mean, self.pw_sigma)
            kl_loss += self.kl_q_p(qb_mean, qb_sigma, self.pb_mean, self.pb_sigma)

        # print(data_loss)
        # print(kl_loss)
        # data_loss = tf.clip_by_value(data_loss, 0)
        loss = minibatch_kl_scaling * kl_loss - data_loss / N

        # Fix it to be like tflearn api


def kl_q_p(q_mean, q_sigma, p_mean, p_sigma):
    log_term = tf.log(p_sigma / (q_sigma + 1e-3))
    mean_term = (tf.square(q_sigma) + tf.square(q_mean - p_mean)) / (2.0 * tf.square(p_sigma) + 1e-3)
    return tf.reduce_sum(log_term + mean_term) - tf.size(q_mean, out_type=tf.float32) * 0.5
