import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted

from skgrf.ensemble import grf
from skgrf.ensemble.base import GRFValidationMixin


class GRFInstrumental(GRFValidationMixin, BaseEstimator):
    r"""GRF Instrumental forest implementation for sci-kit learn.

    Provides a sklearn instrumental interface to the GRF C++ library using Cython.

    :param int n_estimators: The number of tree regressors to train
    :param bool equalize_cluster_weights: Weight the samples such that clusters have
        equally weight. If ``False``, larger clusters will have more weight. If
        ``True``, the number of samples drawn from each cluster is equal to the size of
        the smallest cluster. If ``True``, sample weights should not be passed on
        fitting.
    :param float sample_fraction: Fraction of samples used in each tree. If
        ``ci_group_size`` > 1, the max allowed fraction is 0.5
    :param int mtry: The number of features to split on each node. The default is
        ``sqrt(p) + 20`` where ``p`` is the number of features.
    :param int min_node_size: The minimum number of observations in each tree leaf.
    :param bool honesty: Use honest splitting (subsample splitting).
    :param float honesty_fraction: The fraction of data used for subsample splitting.
    :param bool honesty_prune_leaves: Prune estimation sample tree such that no leaves
        are empty. If ``False``, trees with empty leaves are skipped.
    :param float alpha: The maximum imbalance of a split.
    :param float imbalance_penalty: Penalty applied to imbalanced splits.
    :param int ci_group_size: The quantity of trees grown on each subsample. At least 2
        is required to provide confidence intervals.
    :param double reduced_form_weight: Whether splits should be regularized towards a
        naive splitting criterion that ignores the instrument.
    :param bool stabilize_splits: Whether or not the instrument should be taken into
        account when determining the imbalance of a split.
    :param int n_jobs: The number of threads. Default is number of CPU cores.
    :param int seed: Random seed value.

    :ivar int n_features\_: The number of features (columns) from the fit input ``X``.
    :ivar dict grf_forest\_: The returned result object from calling C++ grf.
    :ivar int mtry\_: The ``mtry`` value determined by validation.
    :ivar int outcome_index\_: The index of the grf train matrix holding the outcomes.
    :ivar list samples_per_cluster\_: The number of samples to train per cluster.
    :ivar list classes\_: The class labels determined from the fit input ``cluster``.
    :ivar int n_classes\_: The number of unique class labels from the fit input
        ``cluster``.
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
        imbalance_penalty=0,
        ci_group_size=2,
        reduced_form_weight=0,
        stabilize_splits=True,
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
        self.imbalance_penalty = imbalance_penalty
        self.ci_group_size = ci_group_size
        self.reduced_form_weight = reduced_form_weight
        self.stabilize_splits = stabilize_splits
        self.n_jobs = n_jobs
        self.seed = seed

    def fit(
        self,
        X,
        y,
        treatment,
        instrument,
        sample_weight=None,
        cluster=None,
    ):
        """Fit the grf forest using training data.

        :param array2d X: training input features
        :param array1d y: training input targets
        :param array1d treatment: training input treatments
        :param array1d instrument: training input instruments
        :param array1d sample_weight: optional weights for input samples
        :param array1d cluster: optional cluster assignments for input samples
        """
        X, y = check_X_y(X, y)
        self.n_features_ = X.shape[1]

        self._check_sample_fraction()
        self._check_alpha()

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)
            use_sample_weights = True
        else:
            use_sample_weights = False

        cluster = self._check_cluster(X=X, cluster=cluster)
        self.samples_per_cluster_ = self._check_equalize_cluster_weights(
            cluster=cluster, sample_weight=sample_weight
        )
        self.mtry_ = self._check_mtry(X=X)
        self._check_reduced_form_weight()

        train_matrix = self._create_train_matrices(
            X=X,
            y=y,
            sample_weight=sample_weight,
            treatment=treatment,
            instrument=instrument,
        )

        self.grf_forest_ = grf.instrumental_train(
            np.asfortranarray(train_matrix.astype("float64")),
            np.asfortranarray([[]]),
            self.outcome_index_,
            self.treatment_index_,
            self.instrument_index_,
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
            self.reduced_form_weight,
            self.alpha,
            self.imbalance_penalty,
            self.stabilize_splits,
            cluster,
            self.samples_per_cluster_,
            False,  # compute_oob_predictions,
            self._get_num_threads(),  # num_threads,
            self.seed,
        )
        return self

    def predict(self, X):
        """Predict regression target for X.

        :param array2d X: prediction input features
        """
        check_is_fitted(self)
        X = check_array(X)

        result = grf.instrumental_predict(
            self.grf_forest_,
            np.asfortranarray([[]]),  # train_matrix
            np.asfortranarray([[]]),  # sparse_train_matrix
            self.outcome_index_,
            self.treatment_index_,
            self.instrument_index_,
            np.asfortranarray(X.astype("float64")),  # test_matrix
            np.asfortranarray([[]]),  # sparse_test_matrix
            self._get_num_threads(),
            False,  # estimate variance
        )
        return np.atleast_1d(np.squeeze(np.array(result["predictions"])))
