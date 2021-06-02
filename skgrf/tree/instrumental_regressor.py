import typing as t

import logging
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted

from skgrf import grf
from skgrf.tree.base import BaseGRFTree
from skgrf.utils.validation import check_sample_weight

if t.TYPE_CHECKING:  # pragma: no cover
    from skgrf.ensemble.instrumental_regressor import GRFInstrumentalRegressor

logger = logging.getLogger(__name__)


class GRFTreeInstrumentalRegressor(BaseGRFTree, RegressorMixin):
    r"""GRF Tree Instrumental regression implementation for sci-kit learn.

    Provides a sklearn tree instrumental regression to the GRF C++ library using Cython.

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
    :param double reduced_form_weight: Whether splits should be regularized towards a
        naive splitting criterion that ignores the instrument.
    :param bool stabilize_splits: Whether or not the instrument should be taken into
        account when determining the imbalance of a split.
    :param int n_jobs: The number of threads. Default is number of CPU cores. Only used
        for target estimation.
    :param int seed: Random seed value.

    :ivar int n_features_in\_: The number of features (columns) from the fit input
        ``X``.
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
        equalize_cluster_weights=False,
        sample_fraction=0.5,
        mtry=None,
        min_node_size=5,
        honesty=True,
        honesty_fraction=0.5,
        honesty_prune_leaves=True,
        alpha=0.05,
        imbalance_penalty=0,
        reduced_form_weight=0,
        stabilize_splits=True,
        n_jobs=-1,
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
        self.reduced_form_weight = reduced_form_weight
        self.stabilize_splits = stabilize_splits
        self.n_jobs = n_jobs
        self.seed = seed

    @classmethod
    def from_forest(cls, forest: "GRFInstrumentalRegressor", idx: int):
        """Extract a tree from a forest.

        :param GRFLocalLinearRegressor forest: A trained GRFLocalLinearRegressor
            instance
        :param int idx: The tree index from the forest to extract.
        """
        # Even though we have a tree object, we keep the exact same dictionary structure
        # that exists in the forests, so that we can reuse the Cython entrypoints.
        # We also copy over some instance attributes from the trained forest.

        # params
        instance = cls(
            equalize_cluster_weights=forest.equalize_cluster_weights,
            sample_fraction=forest.sample_fraction,
            mtry=forest.mtry,
            min_node_size=forest.min_node_size,
            honesty=forest.honesty,
            honesty_fraction=forest.honesty_fraction,
            honesty_prune_leaves=forest.honesty_prune_leaves,
            alpha=forest.alpha,
            imbalance_penalty=forest.imbalance_penalty,
            reduced_form_weight=forest.reduced_form_weight,
            stabilize_splits=forest.stabilize_splits,
            n_jobs=forest.n_jobs,
            seed=forest.seed,
        )
        # forest
        grf_forest = {}
        for k, v in forest.grf_forest_.items():
            if isinstance(v, list):
                grf_forest[k] = [forest.grf_forest_[k][idx]]
            else:
                grf_forest[k] = v
        grf_forest["_num_trees"] = 1
        instance.grf_forest_ = grf_forest
        instance._ensure_ptr()
        # vars
        instance.outcome_index_ = forest.outcome_index_
        instance.treatment_index_ = forest.treatment_index_
        instance.instrument_index_ = forest.instrument_index_
        instance.n_features_in_ = forest.n_features_in_
        instance.classes_ = forest.classes_
        instance.n_classes_ = forest.n_classes_
        instance.samples_per_cluster_ = forest.samples_per_cluster_
        instance.mtry_ = forest.mtry_
        instance.sample_weight_index_ = forest.sample_weight_index_
        return instance

    def fit(
        self,
        X,
        y,
        w,  # treatment
        z,  # instrument
        y_hat=None,
        w_hat=None,
        z_hat=None,
        sample_weight=None,
        cluster=None,
    ):
        """Fit the grf forest using training data.

        :param array2d X: training input features
        :param array1d y: training input targets
        :param array1d w: training input treatments
        :param array1d z: training input instruments
        :param array1d y_hat: estimated expected target responses
        :param array1d w_hat: estimated treatment propensities
        :param array1d z_hat: estimated instrument propensities
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
        self._check_reduced_form_weight()

        if y_hat is None:
            logger.debug("estimating y_hat")
            y_hat = self._estimate_using_regression(
                X=X, y=y, sample_weight=sample_weight, cluster=cluster
            )

        if w_hat is None:
            logger.debug("estimating w_hat")
            w_hat = self._estimate_using_regression(
                X=X, y=w, sample_weight=sample_weight, cluster=cluster
            )

        # don't repeat calculations for causal
        if np.all(w == z):
            z_hat = w_hat

        if z_hat is None:
            logger.debug("estimating z_hat")
            z_hat = self._estimate_using_regression(
                X=X, y=z, sample_weight=sample_weight, cluster=cluster
            )

        y_centered = y - y_hat
        w_centered = w - w_hat
        z_centered = z - z_hat

        train_matrix = self._create_train_matrices(
            X=X,
            y=y_centered,
            sample_weight=sample_weight,
            treatment=w_centered,
            instrument=z_centered,
        )

        self.grf_forest_ = grf.instrumental_train(
            np.asfortranarray(train_matrix.astype("float64")),
            np.asfortranarray([[]]),
            self.outcome_index_,
            self.treatment_index_,
            self.instrument_index_,
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
            self.reduced_form_weight,
            self.alpha,
            self.imbalance_penalty,
            self.stabilize_splits,
            cluster,
            self.samples_per_cluster_,
            False,  # compute_oob_predictions,
            1,  # num_threads,
            self.seed,
        )
        self._ensure_ptr()
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

        result = grf.instrumental_predict(
            self.grf_forest_cpp_,
            np.asfortranarray([[]]),  # train_matrix
            np.asfortranarray([[]]),  # sparse_train_matrix
            self.outcome_index_,
            self.treatment_index_,
            self.instrument_index_,
            np.asfortranarray(X.astype("float64")),  # test_matrix
            np.asfortranarray([[]]),  # sparse_test_matrix
            self._get_num_threads(),
            estimate_variance,
        )
        return result

    def _estimate_using_regression(self, X, y, sample_weight=None, cluster=None):
        """Generate target estimates using a regression.

        In the R package, they perform forest tuning here. For now, we just perform
        a single regression without tuning. We also don't expose any of the forest
        parametrization for this process in the estimator.

        # TODO consider implementing tuning, exposing parameters.
        """
        train_matrix = self._create_train_matrices(
            X=X,
            y=y,
            sample_weight=sample_weight,
        )
        sample_weight, use_sample_weight = check_sample_weight(sample_weight, X)
        n_estimators = 50
        regression_forest = grf.regression_train(
            np.asfortranarray(train_matrix.astype("float64")),
            np.asfortranarray([[]]),
            self.outcome_index_,
            self.sample_weight_index_,
            use_sample_weight,
            self.mtry_,
            n_estimators,  # num_trees
            5,  # min_node_size
            self.sample_fraction,
            True,  # honesty
            0.5,  # honesty_fraction
            self.honesty_prune_leaves,
            1,  # ci_group_size
            self.alpha,
            self.imbalance_penalty,
            cluster,
            self.samples_per_cluster_,
            True,  # compute_oob_predictions,
            self._get_num_threads(),  # num_threads,
            self.seed,
        )
        return np.atleast_1d(np.squeeze(np.array(regression_forest["predictions"])))

    def _more_tags(self):  # pragma: no cover
        return {
            "requires_y": True,
            "_xfail_checks": {
                "check_sample_weights_invariance": "zero sample_weight is not equivalent to removing samples",
            },
        }
