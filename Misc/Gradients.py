import tensorflow as tf


def clip_grads(grads_vars, clip):
    # print("Input:\n", grads_vars)
    grads_vars = list(filter(lambda x: x[0] is not None, grads_vars))
    # print("\nNones removed:\n", grads_vars)
    grads = [g for g, v in grads_vars]
    variables = [v for g, v in grads_vars]
    # print("\nVariables:\n", variables)
    clipped, norm = tf.clip_by_global_norm(grads, clip)
    # print("\nClipped:\n", clipped)
    # print("\nNorm:\n", norm)
    clipped_grads = zip(clipped, variables)
    # print("\nZipped\n:", clipped_grads)
    return clipped_grads, norm
