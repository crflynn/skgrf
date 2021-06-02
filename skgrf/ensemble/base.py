import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from skgrf import grf
from skgrf.base import GRFMixin


class BaseGRFForest(GRFMixin, BaseEstimator):
    def get_split_frequencies(self, max_depth=4):
        """Get the split frequencies of feature indexes at various depths.

        :param int max_depth: The maximum depth of splits to consider
        """
        check_is_fitted(self)
        return np.array(
            grf.compute_split_frequencies(self.grf_forest_cpp_, max_depth=max_depth)
        )

    def get_feature_importances(self, decay_exponent=2, max_depth=4):
        """Get the feature importances.

        :param int decay_exponent: Exponential decay of importance by split depth
        :param int max_depth: The maximum depth of splits to consider
        """
        sf = self.get_split_frequencies(max_depth=max_depth)
        sf = sf / np.maximum(1, np.sum(sf, axis=0))
        weight = np.arange(1.0, sf.shape[0] + 1) ** -decay_exponent
        return np.matmul(sf.T, weight / np.sum(weight))
