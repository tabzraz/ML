import tensorflow as tf

def clip_grads(grads_vars, clip):
    clipped = []
    for grad, var in grads_vars:
        if grad is not None:
            clipped.append((tf.clip_by_norm(grad, clip), var))
    return clipped
