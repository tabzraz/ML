from math import ceil


def tf_conv_size(W, f, s, padding="same"):
    if padding == "same":
        return ceil(W / s)
    else:
        return ceil((W - f + 1) / s)
