import numpy as np
import tensorflow as tf


class Bayesian_Net:

    def __init__(self, layers, likelihood_std=0.1):
        self.likelihood_std = likelihood_std
        self.layers = layers

    def local_reparam_sample(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.local_reparam_sample(input_tensor)

        return input_tensor

    def sample(self, input_tensor, use_mean=False):
        for layer in self.layers:
            input_tensor = layer.sample(input_tensor, use_mean)

        return input_tensor

    def loss(self, input_tensor, target, kl_scaling=1.0, N=4):

        data_loss = 0.0
        for _ in range(N):
            prediction = self.local_reparam_sample(input_tensor)
            data_loss += tf.reduce_sum(log_gaussian_pdf(prediction, target, self.likelihood_std))

        kl_loss = 0.0
        for layer in self.layers:
            kl_loss += layer.kl()

        loss = kl_scaling * kl_loss - data_loss / N

        return loss, kl_scaling * kl_loss, -data_loss / N


def log_gaussian_pdf(x, mu, sigma):
    pdf_val = -(0.5 * tf.log(2 * np.pi) + tf.log(sigma + 1e-5)) - (tf.square(x - mu) / (2 * tf.square(sigma) + 1e-3))
    return pdf_val
