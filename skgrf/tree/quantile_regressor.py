import typing as t

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted

from skgrf import grf
from skgrf.tree.base import BaseGRFTree

if t.TYPE_CHECKING:  # pragma: no cover
    from skgrf.ensemble.quantile_regressor import GRFForestQuantileRegressor


class GRFTreeQuantileRegressor(BaseGRFTree, RegressorMixin):
    r"""GRF Tree Quantile Regression implementation for sci-kit learn.

    Provides a sklearn tree quantile regressor interface to the GRF C++ library using
    Cython.

    .. warning::

        Because the training dataset is required for prediction, the training dataset
        is recorded onto the estimator instance. This means that serializing this
        estimator will result in a file at least as large as the serialized training
        dataset.

    :param list(float) quantiles: A list of quantiles on which to predict.
    :param bool regression_splitting: Use regression splits instead of splitting
        specially for quantiles.
    :param bool equalize_cluster_weights: Weight the samples such that clusters have
        equally weight. If ``False``, larger clusters will have more weight. If
        ``True``, the number of samples drawn from each cluster is equal to the size of
        the smallest cluster. If ``True``, sample weights should not be passed on
        fitting.
    :param float sample_fraction: Fraction of samples used in each tree.
    :param int mtry: The number of features to split on each node. The default is
        ``sqrt(p) + 20`` where ``p`` is the number of features.
    :param int min_node_size: The minimum number of observations in each tree leaf.
    :param bool honesty: Use honest splitting (subsample splitting).
    :param float honesty_fraction: The fraction of data used for subsample splitting.
    :param bool honesty_prune_leaves: Prune estimation sample tree such that no leaves
        are empty. If ``False``, trees with empty leaves are skipped.
    :param float alpha: The maximum imbalance of a split.
    :param float imbalance_penalty: Penalty applied to imbalanced splits.
    :param int seed: Random seed value.

    :ivar int n_features_in\_: The number of features (columns) from the fit input
        ``X``.
    :ivar dict grf_forest\_: The returned result object from calling C++ grf.
    :ivar int mtry\_: The ``mtry`` value determined by validation.
    :ivar int outcome_index\_: The index of the grf train matrix holding the outcomes.
    :ivar list samples_per_cluster\_: The number of samples to train per cluster.
    :ivar list clusters\_: The cluster labels determined from the fit input ``cluster``.
    :ivar int n_clusters\_: The number of unique cluster labels from the fit input
        ``cluster``.
    :ivar array2d train\_: The ``X,y`` concatenated train matrix passed to grf.
    :ivar str criterion: The criterion used for splitting: ``gini``
    """

    def __init__(
        self,
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
        seed=42,
    ):
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
        self.seed = seed

    @property
    def criterion(self):
        return "gini"

    @classmethod
    def from_forest(cls, forest: "GRFForestQuantileRegressor", idx: int):
        """Extract a tree from a forest.

        :param GRFForestQuantileRegressor forest: A trained GRFQuantileRegressor instance
        :param int idx: The tree index from the forest to extract.
        """
        # Even though we have a tree object, we keep the exact same dictionary structure
        # that exists in the forests, so that we can reuse the Cython entrypoints.
        # We also copy over some instance attributes from the trained forest.

        # params
        instance = cls(
            quantiles=forest.quantiles,
            regression_splitting=forest.regression_splitting,
            equalize_cluster_weights=forest.equalize_cluster_weights,
            sample_fraction=forest.sample_fraction,
            mtry=forest.mtry,
            min_node_size=forest.min_node_size,
            honesty=forest.honesty,
            honesty_fraction=forest.honesty_fraction,
            honesty_prune_leaves=forest.honesty_prune_leaves,
            alpha=forest.alpha,
            imbalance_penalty=forest.imbalance_penalty,
            seed=forest.seed,
        )
        # forest
        grf_forest = {}
        for k, v in forest.grf_forest_.items():
            if isinstance(v, list):
                grf_forest[k] = [forest.grf_forest_[k][idx]]
            else:
                grf_forest[k] = v
        grf_forest["num_trees"] = 1
        instance.grf_forest_ = grf_forest
        instance._ensure_ptr()
        # vars
        instance.outcome_index_ = forest.outcome_index_
        instance.n_features_in_ = forest.n_features_in_
        instance.clusters_ = forest.clusters_
        instance.n_clusters_ = forest.n_clusters_
        instance.samples_per_cluster_ = forest.samples_per_cluster_
        instance.mtry_ = forest.mtry_
        instance.sample_weight_index_ = forest.sample_weight_index_
        # data
        instance.train_ = forest.train_
        return instance

    def fit(self, X, y, cluster=None):
        """Fit the grf tree quantile regressor using training data.

        :param array2d X: training input features
        :param array1d y: training input targets
        :param array1d cluster: optional cluster assignments for input samples
        """
        if self.quantiles is None:
            raise ValueError("quantiles must be set")

        X, y = self._validate_data(X, y)
        self._check_num_samples(X)
        self._check_n_features(X, reset=True)

        self._check_sample_fraction()
        self._check_alpha()

        cluster = self._check_cluster(X=X, cluster=cluster)
        self.samples_per_cluster_ = self._check_equalize_cluster_weights(
            cluster=cluster, sample_weight=None
        )
        self.mtry_ = self._check_mtry(X=X)

        train_matrix = self._create_train_matrices(X, y)
        self.train_ = train_matrix

        self.grf_forest_ = grf.quantile_train(
            self.quantiles,
            self.regression_splitting,
            np.asfortranarray(train_matrix.astype("float64")),
            self.outcome_index_,
            self.mtry_,
            1,  # num_trees
            self.min_node_size,
            self.sample_fraction,
            self.honesty,
            self.honesty_fraction,
            self.honesty_prune_leaves,
            1,  # ci_group_size,
            self.alpha,
            self.imbalance_penalty,
            cluster,
            self.samples_per_cluster_,
            False,  # compute_oob_predictions,
            1,  # num_threads
            self.seed,
        )
        self._ensure_ptr()
        sample_weight = np.ones(len(X))
        self._set_sample_weights(sample_weight)
        self._set_node_values(y, sample_weight)
        self._set_n_classes()
        return self

    def predict(self, X):
        """Predict quantile regression target(s) for X.

        :param array2d X: prediction input features
        """
        return np.atleast_1d(np.squeeze(np.array(self._predict(X)["predictions"])))

    def _predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)
        self._ensure_ptr()

        result = grf.quantile_predict(
            self.grf_forest_cpp_,
            self.quantiles,
            np.asfortranarray(self.train_.astype("float64")),
            self.outcome_index_,
            np.asfortranarray(X.astype("float64")),  # test_matrix
            1,  # num_threads
        )
        return result

    def _more_tags(self):
        return {"requires_y": True, "poor_score": True}
