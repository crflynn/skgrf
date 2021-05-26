from sklearn.base import RegressorMixin
from sklearn.tree import BaseDecisionTree
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_array

from skgrf.ensemble.base import GRFValidationMixin
from skgrf.ensemble import grf
from sklearn.base import BaseEstimator

from skgrf.utils.validation import check_sample_weight


class GRFTreeRegressor(GRFValidationMixin, RegressorMixin, BaseEstimator):

    def __init__(
        self,
        equalize_cluster_weights=False,
        sample_fraction=0.5,
        mtry=None,
        min_node_size=5,
        honesty=True,
        honesty_fraction=0.5,
        honesty_prune_leaves=True,
        alpha=0.05,
        imbalance_penalty=0,
        seed=42,
    ):
        self.equalize_cluster_weights = equalize_cluster_weights
        self.sample_fraction = sample_fraction
        self.mtry = mtry
        self.min_node_size = min_node_size
        self.honesty = honesty
        self.honesty_fraction = honesty_fraction
        self.honesty_prune_leaves = honesty_prune_leaves
        self.alpha = alpha
        self.imbalance_penalty = imbalance_penalty
        self.seed = seed
        self.n_jobs = 1

    def fit(
        self, X, y, sample_weight=None, cluster=None, compute_oob_predictions=False
    ):
        """Fit the grf forest using training data.

        :param array2d X: training input features
        :param array1d y: training input targets
        :param array1d sample_weight: optional weights for input samples
        :param array1d cluster: optional cluster assignments for input samples
        """
        X, y = self._validate_data(X, y)
        self._check_num_samples(X)
        self._check_n_features(X, reset=True)

        self._check_sample_fraction(oob=compute_oob_predictions)
        self._check_alpha()

        sample_weight, use_sample_weight = check_sample_weight(sample_weight, X)

        cluster_ = self._check_cluster(X=X, cluster=cluster)
        self.samples_per_cluster_ = self._check_equalize_cluster_weights(
            cluster=cluster_, sample_weight=sample_weight
        )
        self.mtry_ = self._check_mtry(X=X)

        train_matrix = self._create_train_matrices(
            X=X, y=y, sample_weight=sample_weight
        )

        self.grf_forest_ = grf.regression_train(
            np.asfortranarray(train_matrix.astype("float64")),
            np.asfortranarray([[]]),
            self.outcome_index_,
            self.sample_weight_index_,
            use_sample_weight,
            self.mtry_,
            1,  # num_trees
            self.min_node_size,
            self.sample_fraction,
            self.honesty,
            self.honesty_fraction,
            self.honesty_prune_leaves,
            1,  # ci_group_size
            self.alpha,
            self.imbalance_penalty,
            cluster_,
            self.samples_per_cluster_,
            compute_oob_predictions,
            self._get_num_threads(),  # num_threads,
            self.seed,
        )
        return self

    @classmethod
    def from_grf_forest(cls, forest, idx):
        grf_forest_ = forest.grf_forest_
        grf_forest = {}
        for k, v in grf_forest_.items():
            if isinstance(v, list):
                grf_forest[k] = [grf_forest_[k][idx]]
            else:
                grf_forest[k] = v
        grf_forest["_num_trees"] = 1
        instance = cls()
        instance.grf_forest_ = grf_forest
        instance.outcome_index_ = forest.outcome_index_
        instance.n_features_in_ = forest.n_features_in_
        instance.classes_ = forest.classes_
        instance.n_classes_ = forest.n_classes_
        instance.samples_per_cluster_ = forest.samples_per_cluster_
        instance.mtry_ = forest.mtry_
        instance.sample_weight_index_ = forest.sample_weight_index_
        return instance

    def predict(self, X):
        """Predict regression target for X.

        :param array2d X: prediction input features
        """
        return np.atleast_1d(np.squeeze(np.array(self._predict(X)["predictions"])))

    def _predict(self, X, estimate_variance=False):
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)

        result = grf.regression_predict(
            self.grf_forest_,
            np.asfortranarray([[]]),  # train_matrix
            np.asfortranarray([[]]),  # sparse_train_matrix
            self.outcome_index_,
            np.asfortranarray(X.astype("float64")),  # test_matrix
            np.asfortranarray([[]]),  # sparse_test_matrix
            self._get_num_threads(),
            estimate_variance,
        )
        return result
