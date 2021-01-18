import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted

from skgrf.ensemble import grf
from skgrf.ensemble.base import GRFValidationMixin


class GRFQuantileRegressor(GRFValidationMixin, RegressorMixin, BaseEstimator):
    def __init__(
        self,
        n_estimators=100,
        quantiles=None,
        regression_splitting=False,
        equalize_cluster_weights=False,
        sample_fraction=0.5,
        mtry=None,
        min_node_size=5,
        honesty=True,
        honesty_fraction=0.5,
        honesty_prune_leaves=True,
        alpha=0.05,
        imbalance_penalty=0,
        n_jobs=-1,
        seed=42,
    ):
        self.n_estimators = n_estimators
        self.quantiles = quantiles
        self.regression_splitting = regression_splitting
        self.equalize_cluster_weights = equalize_cluster_weights
        self.sample_fraction = sample_fraction
        self.mtry = mtry
        self.min_node_size = min_node_size
        self.honesty = honesty
        self.honesty_fraction = honesty_fraction
        self.honesty_prune_leaves = honesty_prune_leaves
        self.alpha = alpha
        self.imbalance_penalty = imbalance_penalty
        self.n_jobs = n_jobs
        self.seed = seed

    def fit(self, X, y, cluster=None):
        if self.quantiles is None:
            raise ValueError("quantiles must be set")

        X, y = check_X_y(X, y)

        cluster = self._check_cluster(X=X, cluster=cluster)

        samples_per_cluster = self._check_equalize_cluster_weights(cluster=cluster, sample_weight=None)

        if self.mtry is None:
            self.mtry_ = min(np.ceil(np.sqrt(X.shape[1] + 20)), X.shape[1])
        else:
            self.mtry_ = self.mtry

        train_matrix = self._create_train_matrices(X, y)
        self.train_ = train_matrix

        self.grf_forest_ = grf.quantile_train(
            self.quantiles,
            self.regression_splitting,
            np.asfortranarray(train_matrix.astype("float64")),
            np.asfortranarray([[]]),  # sparse_train_matrix
            self.outcome_index_,
            self.mtry_,
            self.n_estimators,  # num_trees
            self.min_node_size,
            self.sample_fraction,
            self.honesty,
            self.honesty_fraction,
            self.honesty_prune_leaves,
            1,  # ci_group_size,
            self.alpha,
            self.imbalance_penalty,
            cluster,
            samples_per_cluster,
            False,  # compute_oob_predictions,
            self._get_num_threads(),  # num_threads
            self.seed,
        )
        print(self.grf_forest_)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        result = grf.quantile_predict(
            self.grf_forest_,
            self.quantiles,
            np.asfortranarray(self.train_.astype("float64")),
            # np.asfortranarray([[]]),  # sparse_train_matrix
            np.asfortranarray([[]]),  # sparse_train_matrix
            self.outcome_index_,
            np.asfortranarray(X.astype("float64")),  # test_matrix
            np.asfortranarray([[]]),  # sparse_test_matrix
            self._get_num_threads(),  # num_threads
        )
        return np.atleast_1d(np.squeeze(np.array(result["predictions"])))
