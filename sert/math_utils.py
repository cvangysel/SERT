import numpy as np
import scipy


def entropy(pk, *args, **kwargs):
    """Proxy for scipy.stats.entropy, with normalized Shannon entropy."""
    if 'normalize' in kwargs:
        normalize = kwargs['normalize']
        del kwargs['normalize']
    else:
        normalize = False

    e = scipy.stats.entropy(pk, *args, **kwargs)

    if normalize:
        num_classes = np.size(pk)
        base = kwargs['base'] if 'base' in kwargs else None

        maximum_entropy = np.log(num_classes)
        if base:
            maximum_entropy /= np.log(base)

        e /= maximum_entropy

    return e
