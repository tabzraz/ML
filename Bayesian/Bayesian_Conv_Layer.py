import numpy as np
import tensorflow as tf


class Bayesian_Conv:

    def __init__(self,
                 # Conv
                 input_channels,
                 filters,
                 filter_height=3,
                 filter_width=3,
                 filter_stride=1,
                 # Variational
                 qw_mean_initial=0.0,
                 qw_std_initial=0.1,
                 qb_mean_initial=0.0,
                 qb_std_initial=0.1,
                 # Prior
                 pw_mean=0.0,
                 pw_sigma=1.0,
                 pb_mean=0.0,
                 pb_sigma=1.0,
                 # Model
                 activation=tf.nn.relu,
                 name="Bayesian_Conv_Layer"
                 ):

        self.input_channels = input_channels
        self.filters = filters
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.filter_stride = filter_stride

        # Hyperparameters
        self.qw_mean_initial = qw_mean_initial
        self.qw_logstd_initial = np.log(np.exp(qw_std_initial) - 1)
        self.qb_mean_initial = qb_mean_initial
        self.qb_logstd_initial = np.log(np.exp(qb_std_initial) - 1)

        # Trainable Variables
        with tf.name_scope(name):
            with tf.name_scope("Weights"):
                qw_mean = tf.Variable(initial_value=np.ones(shape=(filter_height, filter_width, input_channels, filters)) * self.qw_mean_initial, dtype=tf.float32, name="Filter_weight_means")
                qw_p = tf.Variable(initial_value=np.ones(shape=(filter_height, filter_width, input_channels, filters)) * self.qw_logstd_initial, dtype=tf.float32, name="Filter_weight_lodstds")
            with tf.name_scope("Biases"):
                qb_mean = tf.Variable(initial_value=np.ones(shape=(filters,)) * self.qb_mean_initial, dtype=tf.float32, name="Filter_bias_means")
                qb_p = tf.Variable(initial_value=np.ones(shape=(filters,)) * self.qb_logstd_initial, dtype=tf.float32, name="Filter_bias_logstds")

        self.qw_mean = qw_mean
        self.qw_p = qw_p
        self.qw_sigma = tf.nn.softplus(qw_p)
        self.qb_mean = qb_mean
        self.qb_p = qb_p
        self.qb_sigma = tf.nn.softplus(qb_p)

        # Old variational parameters
        self.old_qw_mean = tf.Variable(self.qw_mean.initialized_value())
        self.old_qw_sigma = tf.Variable(tf.nn.softplus(self.qw_p.initialized_value()))
        self.old_qb_mean = tf.Variable(self.qb_mean.initialized_value())
        self.old_qb_sigma = tf.Variable(tf.nn.softplus(self.qb_p.initialized_value()))

        # Baseline variational parameters
        self.baseline_qw_mean = tf.Variable(self.qw_mean.initialized_value())
        self.baseline_qw_p = tf.Variable(self.qw_p.initialized_value())
        self.baseline_qb_mean = tf.Variable(self.qb_mean.initialized_value())
        self.baseline_qb_p = tf.Variable(self.qb_p.initialized_value())

        # Prior hyperparameters
        self.pw_mean = pw_mean
        self.pw_sigma = pw_sigma
        self.pb_mean = pb_mean
        self.pb_sigma = pb_sigma

        # Model
        self.activation = activation

    def sample(self, input_tensor, local_reparam_trick=False, use_mean=False):
        if local_reparam_trick:
            return self.local_reparam_sample(input_tensor)
        else:
            return self.normal_sample(input_tensor, use_mean)

    def normal_sample(self, input_tensor, use_mean=False):
        epsilon_w = tf.random_normal(shape=(self.filter_height, self.filter_width, self.input_channels, self.filters))
        epsilon_b = tf.random_normal(shape=(self.filters, ))

        if use_mean:
            epsilon_w = 0.0
            epsilon_b = 0.0

        W = self.qw_mean + self.qw_sigma * epsilon_w
        b = self.qb_mean + self.qb_sigma * epsilon_b

        conved = tf.nn.conv2d(input_tensor, W, strides=[1, self.filter_stride, self.filter_stride, 1], padding="VALID")
        pre_activation = tf.nn.bias_add(conved, b)
        sample_output = self.activation(pre_activation)

        return sample_output

    def local_reparam_sample(self, input_tensor):
        x = input_tensor
        batch_size = tf.shape(x)[0]

        gamma = tf.nn.conv2d(input_tensor, self.qw_mean, strides=[1, self.filter_stride, self.filter_stride, 1], padding="VALID")
        gamma = tf.nn.bias_add(gamma, self.qb_mean)
        # print(gamma)
        delta = tf.nn.conv2d(tf.square(input_tensor), tf.square(self.qw_sigma), strides=[1, self.filter_stride, self.filter_stride, 1], padding="VALID")
        delta = tf.nn.bias_add(delta, tf.square(self.qb_sigma))
        # print(delta)
        new_height = tf.shape(gamma)[1]
        new_width = tf.shape(gamma)[2]

        epsilon = tf.random_normal(shape=(batch_size, new_height, new_width, self.filters))
        # print(epsilon)

        noisy_activations = gamma + tf.sqrt(delta) * epsilon

        return self.activation(noisy_activations)

    def kl_variational_and_prior(self):
        return kl_q_p(self.qw_mean, self.qw_sigma, self.pw_mean, self.pw_sigma) + kl_q_p(self.qb_mean, self.qb_sigma, self.pb_mean, self.pb_sigma)

    def copy_variational_parameters(self):

        old_new = [(self.old_qw_mean, self.qw_mean),
                   (self.old_qw_sigma, self.qw_sigma),
                   (self.old_qb_mean, self.qb_mean),
                   (self.old_qb_sigma, self.qb_sigma)]

        assigns = [old.assign(new) for old, new in old_new]
        copy_op = tf.group(*assigns)

        return copy_op

    def set_baseline_parameters(self):

        old_new = [(self.baseline_qw_mean, self.qw_mean),
                   (self.baseline_qw_p, self.qw_p),
                   (self.baseline_qb_mean, self.qb_mean),
                   (self.baseline_qb_p, self.qb_p)]

        assigns = [old.assign(new) for old, new in old_new]
        copy_op = tf.group(*assigns)

        return copy_op

    def revert_to_baseline_parameters(self):

        old_new = [(self.baseline_qw_mean, self.qw_mean),
                   (self.baseline_qw_p, self.qw_p),
                   (self.baseline_qb_mean, self.qb_mean),
                   (self.baseline_qb_p, self.qb_p)]

        assigns = [old.assign(new) for new, old in old_new]
        copy_op = tf.group(*assigns)

        return copy_op

    def kl_new_and_old(self):
        kl_w = kl_q_p(self.qw_mean, self.qw_sigma, self.old_qw_mean, self.old_qw_sigma)
        kl_b = kl_q_p(self.qb_mean, self.qb_sigma, self.old_qb_mean, self.old_qb_sigma)

        return kl_w + kl_b


# KL[q|p] where both q and p are fully factorized gaussians
def kl_q_p(q_mean, q_sigma, p_mean, p_sigma):
    log_term = tf.log(p_sigma / (q_sigma + 1e-8))
    mean_term = (tf.square(q_sigma) + tf.square(q_mean - p_mean)) / (2.0 * tf.square(p_sigma) + 1e-8)
    return tf.reduce_sum(log_term + mean_term) - tf.size(q_mean, out_type=tf.float32) * 0.5
