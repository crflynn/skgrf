import multiprocessing
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.tree._tree import csr_matrix
from sklearn.utils.validation import check_array
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

    def get_kernel_weights(self, X, oob_prediction=False):
        """Get training sample weights for test data.

        Given a trained forest and test data,
        compute the kernel weights for each test point.

        Creates a sparse matrix in which the value at (i, j)
        gives the weight of training sample j for test sample i.
        Use ``oob_prediction=True`` if using training set.

        :param array2d X: input features
        :param bool oob_prediction: whether to calculate weights out of bag
        """
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)
        self._ensure_ptr()
        num_threads = self._get_num_threads()
        # this function doesn't accept 0, so we replace 0 with the real cpu count
        if num_threads == 0:
            num_threads = multiprocessing.cpu_count()
        samples, neighbors, weights = grf.compute_kernel_weights(
            self.grf_forest_cpp_,
            np.asfortranarray(X.astype("float64")),
            num_threads,
            oob_prediction,
        )
        return csr_matrix((weights, (samples, neighbors)))
