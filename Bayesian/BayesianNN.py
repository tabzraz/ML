import numpy as np
import tensorflow as tf


class BayesianFC:

    def __init__(self, layer_dims):
        self.layer_dims = layer_dims

        self.qw_means = np.array([])
        self.qw_ps = np.array([])
        self.qb_means = np.array([])
        self.qb_ps = np.array([])
        # Variational Parameters
        for input_dim, output_dim in zip(self.layer_dims[:-1], self.layer_dims[1:]):
            self.qw_means = np.append(self.qw_means, tf.Variable(initial_value=np.zeros(shape=(input_dim, output_dim)), dtype=tf.float32))
            self.qw_ps = np.append(self.qw_ps, tf.Variable(initial_value=np.ones(shape=(input_dim, output_dim)) * 2, dtype=tf.float32))
            self.qb_means = np.append(self.qb_means, tf.Variable(initial_value=np.ones(shape=(output_dim,)) * 0.1, dtype=tf.float32))
            self.qb_ps = np.append(self.qb_ps, tf.Variable(initial_value=np.ones(shape=(output_dim,)) * 2, dtype=tf.float32))

        # Prior Parameters
        # self.pw_mean = tf.constant(value=np.zeros(shape=(input_dim, output_dim)), dtype=tf.float32)
        # self.pw_sigma = tf.constant(value=np.ones(shape=(input_dim, output_dim)) * 1, dtype=tf.float32)
        # self.pb_mean = tf.constant(initial_value=np.ones(shape=(output_dim)) * 0.1, dtype=tf.float32)
        # self.pb_sigma = tf.constant(initial_value=np.ones(shape=(output_dim)) * 1, dtype=tf.float32)
        self.pw_mean = 0.0
        self.pw_sigma = 2.0
        self.pb_mean = 0.0
        self.pb_sigma = 2.0

        # Likelihood std
        # todo: change this to be an output of the network?
        self.likelihood_std = 0.5

        # Weight matrices as variables
        # self.W = tf.Variable(initial_value=np.zeros(shape=(input_dim, output_dim)), dtype=tf.float32)
        # self.b = tf.Variable(initial_value=np.ones(shape=(output_dim,)) * 0.1, dtype=tf.float32)

    def sample(self):
        Ws = []
        bs = []
        sample_input = tf.placeholder(tf.float32, shape=[None, self.layer_dims[0]])

        for qw_mean, qw_p, qb_mean, qb_p, input_dim, output_dim in\
                zip(self.qw_means, self.qw_ps, self.qb_means, self.qb_ps, self.layer_dims[:-1], self.layer_dims[1:]):

            epsilon_w = tf.random_normal(shape=(input_dim, output_dim))
            epsilon_b = tf.random_normal(shape=(output_dim, ))

            W = qw_mean + tf.nn.softplus(qw_p) * epsilon_w
            b = qb_mean + tf.nn.softplus(qb_p) * epsilon_b

            Ws.append(W)
            bs.append(b)

        return sample_input, self.output(sample_input, Ws, bs)

    def output(self, x, Ws, bs):
        for W, b in zip(Ws[:-1], bs[:-1]):
            x = tf.nn.relu(tf.matmul(x, W) + b)

        return tf.matmul(x, Ws[-1]) + bs[-1]
        # return tf.nn.relu(tf.matmul(input, W) + b)

    def gaussian_pdf(self, x, mu, sigma):
        return tf.exp(-(tf.square(x - mu)) / 2 * tf.square(sigma)) / (tf.sqrt(2 * np.pi) * sigma)

    def log_gaussian_clipped_pdf(self, x, mu, sigma):
        pdf_val = (0.5 * tf.log(2 * np.pi) - tf.log(tf.clip_by_value(sigma, 1e-10, 5)) * tf.square(x - mu) / (2 * tf.square(sigma)))
        return tf.clip_by_value(pdf_val, 1e-5, 1.0)

    def calculate_loss(self, data_input, data_target):
        epsilon_ws = []
        epsilon_bs = []
        Ws = []
        bs = []
        log_qw_i = 0
        log_pw_i = 0

        for qw_mean, qw_p, qb_mean, qb_p, input_dim, output_dim in\
                zip(self.qw_means, self.qw_ps, self.qb_means, self.qb_ps, self.layer_dims[:-1], self.layer_dims[1:]):
            # Sample weights
            epsilon_w = tf.random_normal(shape=(input_dim, output_dim))
            epsilon_b = tf.random_normal(shape=(output_dim, ))

            epsilon_ws.append(epsilon_w)
            epsilon_bs.append(epsilon_b)

            W_sigmas = tf.nn.softplus(qw_p)
            b_sigmas = tf.nn.softplus(qb_p)
            W = qw_mean + W_sigmas * epsilon_w
            b = qb_mean + b_sigmas * epsilon_b

            Ws.append(W)
            bs.append(b)

            log_qw_i = tf.reduce_sum(self.log_gaussian_clipped_pdf(W_sigmas * epsilon_w, 0, W_sigmas))
            log_qw_i += tf.reduce_sum(self.log_gaussian_clipped_pdf(b_sigmas * epsilon_b, 0, b_sigmas))

            log_pw_i = tf.reduce_sum(self.log_gaussian_clipped_pdf(W, self.pw_mean, self.pw_sigma))
            log_pw_i += tf.reduce_sum(self.log_gaussian_clipped_pdf(b, self.pb_mean, self.pb_sigma))

        # We assume the nn is a probabilistic model P(y|x,w)
        output = self.output(data_input, Ws, bs)
        log_data_likelihood = tf.reduce_sum(self.log_gaussian_clipped_pdf(data_target, output, self.likelihood_std))

        # todo: divide the first 2 terms by batch size or something
        loss = log_qw_i - log_pw_i - log_data_likelihood

        weight_variables = Ws + bs
        variational_mean_variables = np.concatenate([self.qw_means, self.qb_means])
        variational_std_variables = np.concatenate([self.qw_ps, self.qb_ps])
        variational_std_scaling = epsilon_ws / (1 + np.array([tf.exp(-z) for z in self.qw_ps]))
        variational_std_scaling = np.concatenate([variational_std_scaling, epsilon_bs / (1 + np.array([tf.exp(-z) for z in self.qb_ps]))])
        # + (epsilon_bs / (1 + tf.exp(-self.qb_ps)))

        return loss, weight_variables, variational_mean_variables.tolist(), variational_std_variables.tolist(), variational_std_scaling.tolist()

    def update(self, N, optimiser):
        grads_to_apply = None
        data_input = tf.placeholder(tf.float32, shape=[None, self.layer_dims[0]])
        data_target = tf.placeholder(tf.float32, shape=[None, self.layer_dims[-1]])

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
        return apply_grads_op, data_input, data_target, grads_to_apply, weight_grads, variational_mean_grads_and_vars, variational_std_grads_and_vars
