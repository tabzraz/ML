import numpy as np
import tensorflow as tf


class Bayesian_Net:

    def __init__(self, layers, model_probability):
        self.model_probability = model_probability
        self.layers = layers

    def loss(self, get_output, target, kl_scaling=1.0, N=4, original_prior=True):

        data_loss = 0.0
        for _ in range(N):
            prediction = get_output(local_reparam_trick=True)
            data_loss += tf.reduce_sum(self.model_probability(prediction, target))
            # data_loss += tf.reduce_sum(log_gaussian_pdf(prediction, target, self.likelihood_std))

        kl_loss = 0.0
        for layer in self.layers:
            if original_prior:
                kl_loss += layer.kl_variational_and_prior()
            else:
                kl_loss += layer.kl_new_and_old()

        loss = kl_scaling * kl_loss - data_loss / N

        return loss, kl_scaling * kl_loss, -data_loss / N

    def copy_variational_parameters(self):
        ops = []
        for layer in self.layers:
            op = layer.copy_variational_parameters()
            ops.append(op)

        copy_op = tf.group(*ops)
        return copy_op

    def set_baseline_parameters(self):
        ops = []
        for layer in self.layers:
            op = layer.set_baseline_parameters()
            ops.append(op)

        copy_op = tf.group(*ops)
        return copy_op

    def revert_to_baseline_parameters(self):
        ops = []
        for layer in self.layers:
            op = layer.revert_to_baseline_parameters()
            ops.append(op)

        copy_op = tf.group(*ops)
        return copy_op

    def kl_new_and_old(self):
        kl = 0.0
        for layer in self.layers:
            kl += layer.kl_new_and_old()
        return kl


def log_gaussian_pdf(x, mu, sigma):
    pdf_val = -(0.5 * tf.log(2 * np.pi) + tf.log(sigma + 1e-8)) - (tf.square(x - mu) / (2 * tf.square(sigma) + 1e-8))
    return pdf_val
