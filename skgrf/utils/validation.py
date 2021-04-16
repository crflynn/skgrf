import numpy as np
from sklearn.utils.validation import _check_sample_weight


def check_sample_weight(sample_weight, X):
    if sample_weight is not None:
        sample_weight = _check_sample_weight(sample_weight, X)
        use_sample_weight = True
        if np.array_equal(np.unique(sample_weight), np.array([1.0])):
            sample_weight = None
            use_sample_weight = False
    else:
        use_sample_weight = False

    return sample_weight, use_sample_weight
