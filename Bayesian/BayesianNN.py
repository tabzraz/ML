import numpy as np
import tensorflow as tf


class BayesianFC:

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Variational Parameters
        self.qw_mean = tf.Variable(initial_value=np.zeros(shape=(input_dim, output_dim)), dtype=tf.float32)
        self.qw_p = tf.Variable(initial_value=np.zeros(shape=(input_dim, output_dim)), dtype=tf.float32)
        self.qb_mean = tf.Variable(initial_value=np.ones(shape=(output_dim,)) * 0.1, dtype=tf.float32)
        self.qb_p = tf.Variable(initial_value=np.zeros(shape=(output_dim,)), dtype=tf.float32)

        # Prior Parameters
        # self.pw_mean = tf.constant(value=np.zeros(shape=(input_dim, output_dim)), dtype=tf.float32)
        # self.pw_sigma = tf.constant(value=np.ones(shape=(input_dim, output_dim)) * 1, dtype=tf.float32)
        # self.pb_mean = tf.constant(initial_value=np.ones(shape=(output_dim)) * 0.1, dtype=tf.float32)
        # self.pb_sigma = tf.constant(initial_value=np.ones(shape=(output_dim)) * 1, dtype=tf.float32)
        self.pw_mean = 1.0
        self.pw_sigma = 1.0
        self.pb_mean = 0.0
        self.pb_sigma = 1.0

        # Likelihood std
        # todo: change this to be an output of the network?
        self.likelihood_std = 1.0

        # Weight matrices as variables
        # self.W = tf.Variable(initial_value=np.zeros(shape=(input_dim, output_dim)), dtype=tf.float32)
        # self.b = tf.Variable(initial_value=np.ones(shape=(output_dim,)) * 0.1, dtype=tf.float32)

    def output(self, input, W=None, b=None, sample=False):
        # If we have not provided the weights, then use our distribution to generate them
        if W is None:
            if sample:
                epsilon_w = tf.random_normal(shape=(self.input_dim, self.output_dim))
                epsilon_b = tf.random_normal(shape=(self.output_dim, ))
            else:
                epsilon_w = epsilon_b = 0

            W = self.qw_mean + tf.nn.softplus(self.qw_p) * epsilon_w
            b = self.qb_mean + tf.nn.softplus(self.qb_p) * epsilon_b

        # Now we have the weights, calculate the output of the network
        return tf.matmul(input, W) + b
        # return tf.nn.relu(tf.matmul(input, W) + b)

    def gaussian_pdf(self, x, mu, sigma):
        return tf.exp(-(tf.square(x - mu)) / 2 * tf.square(sigma)) / (tf.sqrt(2 * np.pi) * sigma)

    def calculate_loss(self, data_input, data_target):
        # Sample weights
        epsilon_w = tf.random_normal(shape=(self.input_dim, self.output_dim))
        epsilon_b = tf.random_normal(shape=(self.output_dim, ))

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
        # We assume the nn is a probabilistic model P(y|x,w)
        output = self.output(data_input, W=W, b=b, sample=False)
        data_likelihood = tf.reduce_sum(self.gaussian_pdf(data_target, output, self.likelihood_std))

        log_data_likelihood = tf.log(data_likelihood)

        loss = log_qw_i - log_pw_i - log_data_likelihood

        weight_variables = [W, b]
        variational_mean_variables = [self.qw_mean, self.qb_mean]
        variational_std_variables = [self.qw_p, self.qb_p]
        variational_std_scaling = [epsilon_w / (1 + tf.exp(-self.qw_p)), epsilon_b / (1 + tf.exp(-self.qb_p))]

        return loss, weight_variables, variational_mean_variables, variational_std_variables, variational_std_scaling

    def update(self, N, optimiser):
        grads_to_apply = None
        data_input = tf.placeholder(tf.float32, shape=[None, self.input_dim])
        data_target = tf.placeholder(tf.float32, shape=[None, self.output_dim])

        for _ in range(N):
            loss, w_vars, v_m_vars, v_std_vars, v_std_scaling = self.calculate_loss(data_input, data_target)

            # weight_grads_and_vars = optimiser.compute_gradients(loss, var_list=w_vars)
            weight_grads = tf.gradients(loss, w_vars)
            # print(weight_grads)
            variational_mean_grads_and_vars = optimiser.compute_gradients(loss, var_list=v_m_vars)
            # print(variational_mean_grads_and_vars)
            variational_std_grads_and_vars = optimiser.compute_gradients(loss, var_list=v_std_vars)
            # print(variational_std_grads_and_vars)

            new_variational_mean_grads_and_vars = [(gw + gv, v) for gw, (gv, v) in zip(weight_grads, variational_mean_grads_and_vars)]
            new_variational_std_grads_and_vars = [(gw * gws + gv, v) for gw, (gv, v), gws in zip(weight_grads, variational_std_grads_and_vars, v_std_scaling)]
            concat_grads = new_variational_mean_grads_and_vars + new_variational_std_grads_and_vars

            if grads_to_apply is None:
                grads_to_apply = concat_grads
            else:
                grads_to_apply = [(g + ug, v) for (g, v), (ug, _) in zip(grads_to_apply, concat_grads)]

        apply_grads_op = optimiser.apply_gradients(grads_to_apply)
        return apply_grads_op, data_input, data_target, grads_to_apply
