import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.utils import check_X_y
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted

from skgrf.ensemble import grf
from skgrf.ensemble.base import GRFValidationMixin


class GRFRegressor(GRFValidationMixin, RegressorMixin, BaseEstimator):
    def __init__(
        self,
        n_estimators=100,
        equalize_cluster_weights=False,
        sample_fraction=0.5,
        mtry=None,
        min_node_size=5,
        honesty=True,
        honesty_fraction=0.5,
        honesty_prune_leaves=True,
        alpha=0.05,
        imbalance_penalty=0,
        ci_group_size=2,
        tune_num_trees=50,
        tune_num_reps=100,
        tune_num_draws=1000,
        compute_oob_predictions=True,  # TODO remove?
        n_jobs=0,
        seed=42,
    ):
        self.n_estimators = n_estimators
        self.equalize_cluster_weights = equalize_cluster_weights
        self.sample_fraction = sample_fraction
        self.mtry = mtry
        self.min_node_size = min_node_size
        self.honesty = honesty
        self.honesty_fraction = honesty_fraction
        self.honesty_prune_leaves = honesty_prune_leaves
        self.alpha = alpha
        self.imbalance_penalty = imbalance_penalty
        self.ci_group_size = ci_group_size
        self.tune_num_trees = tune_num_trees
        self.tune_num_reps = tune_num_reps
        self.tune_num_draws = tune_num_draws
        self.compute_oob_predictions = compute_oob_predictions
        self.n_jobs = n_jobs
        self.seed = seed

    def fit(self, X, y, sample_weight=None, cluster=None):
        X, y = check_X_y(X, y)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        cluster = self._check_cluster(cluster, sample_weight)

        samples_per_cluster = self._check_equalize_cluster_weights(cluster, sample_weight)

        if self.mtry is None:
            self.mtry_ = min(np.ceil(np.sqrt(X.shape[1] + 20)), X.shape[1])
        else:
            self.mtry_ = self.mtry

        if sample_weight is None:
            use_sample_weights = False
        else:
            use_sample_weights = True

        self._check_n_jobs()

        train_matrix = self._create_train_matrices(X, y, sample_weight=sample_weight)

        self.grf_forest_ = grf.regression_train(
            np.asfortranarray(train_matrix.astype("float64")),
            np.asfortranarray([[]]),
            self.outcome_index_,
            self.sample_weight_index_,
            use_sample_weights,
            self.mtry_,
            self.n_estimators,  # num_trees
            self.min_node_size,
            self.sample_fraction,
            self.honesty,
            self.honesty_fraction,
            self.honesty_prune_leaves,
            self.ci_group_size,
            self.alpha,
            self.imbalance_penalty,
            cluster,
            samples_per_cluster,
            self.compute_oob_predictions,
            self.n_jobs,  # num_threads,
            self.seed,
        )
        print(self.grf_forest_)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        result = grf.regression_predict(
            self.grf_forest_,
            np.asfortranarray([[]]),  # train_matrix
            np.asfortranarray([[]]),  # sparse_train_matrix
            self.outcome_index_,
            np.asfortranarray(X.astype("float64")),  # test_matrix
            np.asfortranarray([[]]),  # sparse_test_matrix
            self.n_jobs,
            0,  # estimate variance
        )
        return np.atleast_1d(np.squeeze(np.array(result["predictions"])))
