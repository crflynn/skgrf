import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted

from skgrf import grf
from skgrf.ensemble.base import BaseGRFForest
from skgrf.tree.survival import GRFTreeSurvival
from skgrf.utils.validation import check_sample_weight


class GRFForestSurvival(BaseGRFForest, BaseEstimator):
    r"""GRF Survival implementation for sci-kit learn.

    Provides a sklearn survival interface to the GRF C++ library using Cython.

    .. warning::

        Because the training dataset is required for prediction, the training dataset
        is recorded onto the estimator instance. This means that serializing this
        estimator will result in a file at least as large as the serialized training
        dataset.

    :param int n_estimators: The number of survival trees to train
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
    :param int n_jobs: The number of threads. Default is number of CPU cores.
    :param int seed: Random seed value.

    :ivar list estimators\_: A list of tree objects from the forest.
    :ivar int n_features_in\_: The number of features (columns) from the fit input
        ``X``.
    :ivar dict grf_forest\_: The returned result object from calling C++ grf.
    :ivar int mtry\_: The ``mtry`` value determined by validation.
    :ivar int outcome_index\_: The index of the grf train matrix holding the outcomes.
    :ivar int censor_index\_: The index of the grf train matrix holding the censoring.
    :ivar array1d failure_times_\_: An array of unique failure times from the training
        set.
    :ivar int num_failures_\_: The length of the ``failure_times`` array.
    :ivar list clusters\_: The cluster labels determined from the fit input ``cluster``.
    :ivar int n_clusters\_: The number of unique cluster labels from the fit input
        ``cluster``.
    :ivar str criterion: The criterion used for splitting: ``logrank``
    """

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
        n_jobs=-1,
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
        self.n_jobs = n_jobs
        self.seed = seed

    @property
    def criterion(self):
        return "logrank"

    @property
    def estimators_(self):
        try:
            check_is_fitted(self)
        except NotFittedError:
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute 'estimators_'"
            ) from None
        return [
            GRFTreeSurvival.from_forest(self, idx=idx)
            for idx in range(self.n_estimators)
        ]

    def get_estimator(self, idx):
        """Extract a single estimator tree from the forest.

        :param int idx: The index of the tree to extract.
        """
        check_is_fitted(self)
        return GRFTreeSurvival.from_forest(self, idx=idx)

    def fit(self, X, y, sample_weight=None, cluster=None):
        """Fit the grf forest using training data.

        :param array2d X: training input features
        :param array1d y: training input targets, rows of (bool, float) representing
            (survival, time)
        :param array1d sample_weight: optional weights for input samples
        :param array1d cluster: optional cluster assignments for input samples
        """
        X = check_array(X)
        self._check_num_samples(X)
        self._check_n_features(X, reset=True)
        y = np.array(y.tolist())

        self._check_sample_fraction()
        self._check_alpha()

        cluster = self._check_cluster(X=X, cluster=cluster)
        self.samples_per_cluster_ = self._check_equalize_cluster_weights(
            cluster=cluster, sample_weight=sample_weight
        )

        sample_weight, use_sample_weight = check_sample_weight(sample_weight, X)

        self.mtry_ = self._check_mtry(X=X)

        # Extract the failure times from the training targets
        self.failure_times_ = np.sort(np.unique(y[:, 1][y[:, 0] == 1]))
        self.num_failures_ = len(self.failure_times_)

        # Relabel the failure times to consecutive integers
        y_times_relabeled = np.searchsorted(self.failure_times_, y[:, 1])
        y_censor = y[:, 0]

        train_matrix = self._create_train_matrices(
            X, y_times_relabeled, sample_weight=sample_weight, censor=y_censor
        )
        self.train_ = train_matrix

        self.grf_forest_ = grf.survival_train(
            np.asfortranarray(train_matrix.astype("float64")),
            self.outcome_index_,
            self.censor_index_,
            self.sample_weight_index_,
            use_sample_weight,
            self.mtry_,
            self.n_estimators,  # num_trees
            self.min_node_size,
            self.sample_fraction,
            self.honesty,
            self.honesty_fraction,
            self.honesty_prune_leaves,
            self.alpha,
            self.num_failures_,
            cluster,
            self.samples_per_cluster_,
            False,  # compute_oob_predictions,
            self._get_num_threads(),  # num_threads,
            self.seed,
        )
        self._ensure_ptr()
        sample_weight = sample_weight if sample_weight is not None else np.ones(len(X))
        self._set_sample_weights(sample_weight)
        self._set_node_values(y, sample_weight)
        self._set_n_classes()
        return self

    def predict_cumulative_hazard_function(self, X):
        """Predict cumulative hazard function.

        :param array2d X: prediction input features
        """
        surv = self.predict_survival_function(X)
        return -np.log(surv)

    def predict(self, X):
        """Predict risk score.

        :param array2d X: prediction input features
        """
        chf = self.predict_cumulative_hazard_function(X)
        return chf.sum(1)

    def predict_survival_function(self, X):
        """Predict survival function.

        :param array2d X: prediction input features
        """
        return np.atleast_1d(np.squeeze(np.array(self._predict(X)["predictions"])))

    def _predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)
        self._ensure_ptr()

        result = grf.survival_predict(
            self.grf_forest_cpp_,
            np.asfortranarray(self.train_.astype("float64")),  # test_matrix
            self.outcome_index_,
            self.censor_index_,
            self.sample_weight_index_,
            False,  # use_sample_weights
            np.asfortranarray(X.astype("float64")),  # test_matrix
            self._get_num_threads(),
            self.num_failures_,
        )
        return result

    def _more_tags(self):
        return {
            "requires_y": True,
            "_xfail_checks": {
                "check_sample_weights_invariance": "zero sample_weight is not equivalent to removing samples",
            },
        }
