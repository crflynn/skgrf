import typing as t

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted

from skgrf import grf
from skgrf.tree.base import BaseGRFTree
from skgrf.utils.validation import check_sample_weight

if t.TYPE_CHECKING:  # pragma: no cover
    from skgrf.ensemble.local_linear_regressor import GRFForestLocalLinearRegressor


class GRFTreeLocalLinearRegressor(BaseGRFTree, RegressorMixin):
    r"""GRF Tree Local Linear Regression implementation for sci-kit learn.

    Provides a sklearn tree regressor interface to the GRF C++ library using Cython.

    .. warning::

        Because the training dataset is required for prediction, the training dataset
        is recorded onto the estimator instance. This means that serializing this
        estimator will result in a file at least as large as the serialized training
        dataset.

    :param bool ll_split_weight_penalty: Use a covariance ridge penalty if using local
        linear splits.
    :param float ll_split_lambda: Ridge penalty for splitting.
    :param list(int) ll_split_variables: Linear correction variables for splitting. Uses
        all variables if not specified.
    :param float ll_split_cutoff: Leaf size after which the overall beta is used. If
        unspecified, default is sqrt of num samples. Passing 0 means no cutoff.
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
    :ivar list clusters\_: The cluster labels determined from the fit input ``cluster``.
    :ivar int n_clusters\_: The number of unique cluster labels from the fit input
        ``cluster``.
    :ivar array2d train\_: The ``X,y`` concatenated train matrix passed to grf.
    :ivar str criterion: The criterion used for splitting: ``mse``
    """

    def __init__(
        self,
        ll_split_weight_penalty=False,
        ll_split_lambda=0.1,
        ll_split_variables=None,
        ll_split_cutoff=None,
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
        self.ll_split_weight_penalty = ll_split_weight_penalty
        self.ll_split_lambda = ll_split_lambda
        self.ll_split_variables = ll_split_variables
        self.ll_split_cutoff = ll_split_cutoff
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
        return "mse"

    @classmethod
    def from_forest(cls, forest: "GRFForestLocalLinearRegressor", idx: int):
        """Extract a tree from a forest.

        :param GRFForestLocalLinearRegressor forest: A trained GRFLocalLinearRegressor
            instance
        :param int idx: The tree index from the forest to extract.
        """
        # Even though we have a tree object, we keep the exact same dictionary structure
        # that exists in the forests, so that we can reuse the Cython entrypoints.
        # We also copy over some instance attributes from the trained forest.

        # params
        instance = cls(
            ll_split_weight_penalty=forest.ll_split_weight_penalty,
            ll_split_lambda=forest.ll_split_lambda,
            ll_split_variables=forest.ll_split_variables,
            ll_split_cutoff=forest.ll_split_cutoff,
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
        instance.ll_split_variables_ = forest.ll_split_variables_
        instance.ll_split_cutoff_ = forest.ll_split_cutoff_
        instance.overall_beta_ = forest.overall_beta_
        # data
        instance.train_ = forest.train_
        return instance

    def fit(self, X, y, sample_weight=None, cluster=None):
        """Fit the grf forest using training data.

        :param array2d X: training input features
        :param array1d y: training input targets
        :param array1d sample_weight: optional weights for input samples
        :param array1d cluster: optional cluster assignments for input samples
        """
        X, y = self._validate_data(X, y)
        self._check_num_samples(X)
        self._check_n_features(X, reset=True)

        self._check_sample_fraction()
        self._check_alpha()

        sample_weight, use_sample_weight = check_sample_weight(sample_weight, X)

        cluster = self._check_cluster(X=X, cluster=cluster)
        self.samples_per_cluster_ = self._check_equalize_cluster_weights(
            cluster=cluster, sample_weight=sample_weight
        )
        self.mtry_ = self._check_mtry(X=X)

        train_matrix = self._create_train_matrices(X, y, sample_weight=sample_weight)
        self.train_ = train_matrix

        if self.ll_split_variables is None:
            self.ll_split_variables_ = list(range(X.shape[1]))
        else:
            self.ll_split_variables_ = self.ll_split_variables

        # calculate overall beta
        if self.ll_split_cutoff is None:
            self.ll_split_cutoff_ = int(X.shape[0] ** 0.5)
        else:
            self.ll_split_cutoff_ = self.ll_split_cutoff

        if self.ll_split_cutoff_ > 0:
            J = np.eye(X.shape[1] + 1)
            J[0, 0] = 0
            D = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
            self.overall_beta_ = (
                np.linalg.solve(
                    D.T @ D + self.ll_split_lambda * J, np.eye(X.shape[1] + 1)
                )
                @ D.T
                @ y
            )
        else:
            self.overall_beta_ = np.empty((0,), dtype=float, order="F")

        self.grf_forest_ = grf.ll_regression_train(
            np.asfortranarray(train_matrix.astype("float64")),
            self.outcome_index_,
            self.sample_weight_index_,
            self.ll_split_lambda,
            self.ll_split_weight_penalty,
            self.ll_split_variables_,
            self.ll_split_cutoff_,
            self.overall_beta_,
            use_sample_weight,
            self.mtry_,
            1,  # num_trees
            self.min_node_size,
            self.sample_fraction,
            self.honesty,
            self.honesty_fraction,
            self.honesty_prune_leaves,
            1,
            self.alpha,
            self.imbalance_penalty,
            cluster,
            self.samples_per_cluster_,
            1,  # num_threads,
            self.seed,
        )
        self._ensure_ptr()
        sample_weight = sample_weight if sample_weight is not None else np.ones(len(X))
        self._set_sample_weights(sample_weight)
        self._set_node_values(y, sample_weight)
        self._set_n_classes()
        return self

    def predict(self, X):
        """Predict regression target for X.

        :param array2d X: prediction input features
        """
        return np.atleast_1d(np.squeeze(np.array(self._predict(X)["predictions"])))

    def _predict(self, X, estimate_variance=False):
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)
        self._ensure_ptr()

        result = grf.ll_regression_predict(
            self.grf_forest_cpp_,
            np.asfortranarray(self.train_.astype("float64")),  # train_matrix
            self.outcome_index_,
            np.asfortranarray(X.astype("float64")),  # test_matrix
            [self.ll_split_lambda],  # ll_lambda
            self.ll_split_weight_penalty,  # ll_weight_penalty
            self.ll_split_variables_,  # linear_correction_variables
            1,  # num_threads
            estimate_variance,  # estimate variance
        )
        return result

    def _more_tags(self):
        return {
            "requires_y": True,
            "_xfail_checks": {
                "check_sample_weights_invariance": "zero sample_weight is not equivalent to removing samples",
            },
        }
