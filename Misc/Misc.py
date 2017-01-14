from math import ceil

def tf_conv_size(W, f, s):
    return ceil((W - f + 1) / s)
