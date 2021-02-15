import numpy as np
from abc import ABC
from abc import abstractmethod
from scipy import stats as ss
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import check_X_y
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted

from skgrf.ensemble import GRFRegressor
from skgrf.ensemble import grf
from skgrf.ensemble.base import GRFValidationMixin


class GRFBoostedRegressor(GRFValidationMixin, RegressorMixin, BaseEstimator):
    r"""GRF Boosted Regression implementation for sci-kit learn.

    Provides a sklearn regressor interface to the GRF C++ library using Cython.

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
        tune_params="none",
        tune_n_estimators=10,
        tune_n_reps=100,
        tune_n_draws=1000,
        boost_steps=None,
        boost_error_reduction=0.97,
        boost_max_steps=5,
        boost_trees_tune=10,
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
        self.tune_params = tune_params
        self.tune_n_estimators = tune_n_estimators
        self.tune_n_reps = tune_n_reps
        self.tune_n_draws = tune_n_draws
        self.boost_steps = boost_steps
        self.boost_error_reduction = boost_error_reduction
        self.boost_max_steps = boost_max_steps
        self.boos_trees_tune = boost_trees_tune
        self.n_jobs = n_jobs
        self.seed = seed

    def fit(self, X, y, sample_weight=None, cluster=None):
        """Fit the grf forest using training data.

        :param array2d X: training input features
        :param array1d y: training input targets
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

        train_matrix = self._create_train_matrices(
            X=X, y=y, sample_weight=sample_weight
        )

        regression_forest = GRFRegressor(
            n_estimators=self.n_estimators,
            equalize_cluster_weights=self.equalize_cluster_weights,
            sample_fraction=self.sample_fraction,
            mtry=self.mtry,
            min_node_size=self.min_node_size,
            honesty=self.honesty,
            honesty_fraction=self.honesty_fraction,
            honesty_prune_leaves=self.honesty_prune_leaves,
            alpha=self.alpha,
            imbalance_penalty=self.imbalance_penalty,
            ci_group_size=self.ci_group_size,
            n_jobs=self.n_jobs,
            seed=self.seed,
        )
        tunable_params = (
            "sample_fraction",
            "mtry",
            "min_node_size",
            "honesty_fraction",
            "honesty_prune_leaves",
            "alpha",
            "imbalance_penalty",
        )
        param_distributions = {}
        for param in self.tune_params:
            if param not in tunable_params:
                raise ValueError(
                    f"tuning param {param} not found in {str(tunable_params)}"
                )
            param_distributions[param] = PARAM_DISTRIBUTIONS[param](*X.shape)

        # for k in range()
        rscv = RandomizedSearchCV(
            estimator=regression_forest, param_distributions=param_distributions
        )
        rscv.fit(X=X, y=y)
        # rscv.

        # rsc
        # here we do the tuning / randomized cv

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
        return np.atleast_1d(np.squeeze(np.array(self._predict(X)["predictions"])))

    def _predict(self, X, estimate_variance=False):
        check_is_fitted(self)
        X = check_array(X)

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


class GRFParamDistribution(ABC):
    def __init__(self, X_rows: int, X_cols: int):
        self.X_rows = X_rows
        self.X_cols = X_cols

    @abstractmethod
    def rvs(self, *args, **kwds):
        raise NotImplementedError()


class GRFMinNodeSizeDistribution(GRFParamDistribution):
    def rvs(self, *args, **kwds):
        return np.floor(
            np.exp2(ss.uniform(*args, **kwds) * (np.log(self.X_rows) - np.log(2) - 4))
        )


class GRFSampleFractionDistribution(GRFParamDistribution):
    def rvs(self, *args, **kwds):
        return 0.05 + 0.45 * ss.uniform(*args, **kwds)


class GRFMtryDistribution(GRFParamDistribution):
    def rvs(self, *args, **kwds):
        return np.ceil(
            np.min([self.X_cols, np.sqrt(self.X_cols) + 20]) * ss.uniform(*args, **kwds)
        )


class GRFAlphaDistribution(GRFParamDistribution):
    def rvs(self, *args, **kwds):
        return ss.uniform(*args, **kwds) / 4


class GRFImbalancePenaltyDistribution(GRFParamDistribution):
    def rvs(self, *args, **kwds):
        return -np.log(ss.uniform(*args, **kwds))


class GRFHonestyFractionDistribution(GRFParamDistribution):
    def rvs(self, *args, **kwds):
        return 0.5 + 0.3 * ss.uniform(*args, **kwds)


class GRFHonestyPruneLeavesDistribution(GRFParamDistribution):
    def __init__(self, X_rows: int, X_cols: int):
        super().__init__(X_rows, X_cols)
        self.choices = np.array([True, False])

    def rvs(self, *args, **kwds):
        return self.choices[ss.bernoulli(*args, **kwds)]


PARAM_DISTRIBUTIONS = {
    "min_node_size": GRFMinNodeSizeDistribution,
    "sample_fraction": GRFSampleFractionDistribution,
    "mtry": GRFMtryDistribution,
    "alpha": GRFAlphaDistribution,
    "imbalance_penalty": GRFImbalancePenaltyDistribution,
    "honesty_fraction": GRFHonestyFractionDistribution,
    "honesty_prune_leaves": GRFHonestyPruneLeavesDistribution,
}
