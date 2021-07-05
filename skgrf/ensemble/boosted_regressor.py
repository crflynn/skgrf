import logging
import numpy as np
from abc import ABC
from abc import abstractmethod
from numpy import random
from scipy import stats as ss
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted

from skgrf import grf
from skgrf.base import GRFMixin
from skgrf.ensemble.regressor import GRFForestRegressor
from skgrf.utils.validation import check_sample_weight

logger = logging.getLogger(__name__)


class GRFBoostedForestRegressor(GRFMixin, RegressorMixin, BaseEstimator):
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
    :param list(str) tune_params: A list of parameter names on which to perform tuning.
        Valid strings are "sample_fraction", "mtry", "min_node_size",
        "honesty_fraction", "honesty_prune_leaves", "alpha", "imbalance_penalty".
    :param int tune_n_estimators: The number of estimators to use in the tuning model.
    :param int tune_n_reps: The number of forests used in the tuning model
    :param int tune_n_draws: The number of random parameter values for tuning model
        selection
    :param int boost_steps: The number of boosting iterations
    :param int boost_error_reduction: The percentage of previous step's error that
        must be estimated by the current boost step
    :param int boost_max_steps: The maximum number of boosting iterations
    :param int boost_trees_tune: The number of trees used to test a new boosting step.
    :param int n_jobs: The number of threads. Default is number of CPU cores.
    :param int seed: Random seed value.

    :ivar int n_features_in\_: The number of features (columns) from the fit input
        ``X``.
    :ivar dict boosted_forests\_: The boosted regression forests.
    :ivar int mtry\_: The ``mtry`` value determined by validation.
    :ivar int outcome_index\_: The index of the grf train matrix holding the outcomes.
    :ivar list samples_per_cluster\_: The number of samples to train per cluster.
    :ivar list clusters\_: The cluster labels determined from the fit input ``cluster``.
    :ivar int n_clusters\_: The number of unique cluster labels from the fit input
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
        tune_params=None,
        tune_n_estimators=50,
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
        self.boost_trees_tune = boost_trees_tune
        self.n_jobs = n_jobs
        self.seed = seed

    def fit(self, X, y, sample_weight=None, cluster=None):
        """Fit the grf forest using training data.

        :param array2d X: training input features
        :param array1d y: training input targets
        :param array1d sample_weight: optional weights for input samples
        :param array1d cluster: optional cluster assignments for input samples
        """
        X, y = self._validate_data(X, y)
        self._check_n_features(X, reset=True)

        self._check_boost_error_reduction()

        self._check_sample_fraction(oob=True)
        self._check_alpha()

        sample_weight, _ = check_sample_weight(sample_weight, X)

        cluster_ = self._check_cluster(X=X, cluster=cluster)
        self.samples_per_cluster_ = self._check_equalize_cluster_weights(
            cluster=cluster_, sample_weight=sample_weight
        )
        self.mtry_ = self._check_mtry(X=X)

        _ = self._create_train_matrices(X=X, y=y, sample_weight=sample_weight)

        # region tuning a regression forest
        regression_forest = GRFForestRegressor(
            n_estimators=self.tune_n_estimators,
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
        if self.tune_params is None:
            logger.debug("not tuning boosted forest")
            regression_forest.fit(
                X=X,
                y=y,
                sample_weight=sample_weight,
                cluster=cluster,
                compute_oob_predictions=True,
            )
            params = regression_forest.get_params(deep=True)
            forest = regression_forest
        else:
            logger.debug("tuning boosted forest")
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

            uniform_samples = random.uniform(
                size=(self.tune_n_draws, len(self.tune_params))
            )
            param_samples = np.zeros(shape=(self.tune_n_draws, len(self.tune_params)))
            for idx, param in enumerate(self.tune_params):
                param_samples[:, idx] = param_distributions[param].dist(
                    uniform_samples[:, idx]
                )

            errors = []
            for draw in range(self.tune_n_draws):
                params = {
                    p: param_samples[draw, idx]
                    for idx, p in enumerate(self.tune_params)
                }
                regression_forest.set_params(**params)
                regression_forest.fit(
                    X=X,
                    y=y,
                    sample_weight=sample_weight,
                    cluster=cluster,
                    compute_oob_predictions=True,
                )
                errors.append(
                    np.nanmean(regression_forest.grf_forest_["debiased_error"])
                )

            if np.any(np.isnan(errors)):
                raise ValueError(
                    "unable to tune because of NaN-valued forest error estimates; consider more trees"
                )

            if np.std(errors) == 0 or np.std(errors) / np.mean(errors) < 1e-10:
                raise ValueError(
                    "unable to tune because of constant errors for forests; consider more trees"
                )

            variance_guess = np.var(errors) / 2
            gp = GaussianProcessRegressor(alpha=variance_guess)
            gp.fit(uniform_samples, errors)

            opt_samples = random.uniform(
                size=(self.tune_n_draws, len(self.tune_params))
            )

            model_surface = gp.predict(opt_samples)
            tuned_params = np.zeros(shape=(self.tune_n_draws, len(self.tune_params)))
            for idx, param in enumerate(self.tune_params):
                tuned_params[:, idx] = param_distributions[param].dist(
                    opt_samples[:, idx]
                )

            opt_idx = np.argmin(model_surface)
            params = {
                p: tuned_params[opt_idx, idx] for idx, p in enumerate(self.tune_params)
            }
            params.update(**{"n_estimators": self.tune_n_estimators * 4})
            regression_forest.set_params(**params)
            regression_forest.fit(
                X,
                y,
                sample_weight=sample_weight,
                cluster=cluster,
                compute_oob_predictions=True,
            )
            retrained_error = np.nanmean(
                regression_forest.grf_forest_["debiased_error"]
            )

            default_params = {
                "sample_fraction": 0.5,
                "mtry": min(np.ceil(np.sqrt(X.shape[1]) + 20), X.shape[1]),
                "min_node_size": 5,
                "honesty_fraction": 0.5,
                "honesty_prune_leaves": True,
                "alpha": 0.05,
                "imbalance_penalty": 0,
            }
            default_forest = clone(regression_forest)
            default_forest.set_params(**default_params)
            default_forest.fit(
                X=X,
                y=y,
                sample_weight=sample_weight,
                cluster=cluster,
                compute_oob_predictions=True,
            )
            default_error = np.nanmean(default_forest.grf_forest_["debiased_error"])

            if default_error < retrained_error:
                params = default_forest.get_params()
                forest = default_forest
            else:
                params = regression_forest.get_params()
                forest = regression_forest
        # endregion

        # region boosting with the tuned forest
        logger.debug("boosting forest")
        current_pred = {
            "predictions": forest.grf_forest_["predictions"],
            "debiased_error": forest.grf_forest_["debiased_error"],
            "excess_error": forest.grf_forest_["excess_error"],
        }

        y_hat = np.atleast_1d(np.squeeze(np.array(current_pred["predictions"])))
        debiased_error = current_pred["debiased_error"]
        boosted_forests = {
            "forest": [forest],
            "error": [np.mean(debiased_error)],
        }

        step = 1
        while True:
            y_residual = y - y_hat
            if self.boost_steps is not None:
                if step > self.boost_steps:
                    break
            elif step > self.boost_max_steps:
                break
            else:
                forest_small = GRFForestRegressor(
                    sample_fraction=params["sample_fraction"],
                    mtry=params["mtry"],
                    n_estimators=self.boost_trees_tune,
                    n_jobs=self.n_jobs,
                    min_node_size=params["min_node_size"],
                    honesty=self.honesty,
                    honesty_fraction=params["honesty_fraction"],
                    honesty_prune_leaves=params["honesty_prune_leaves"],
                    seed=self.seed,
                    ci_group_size=self.ci_group_size,
                    alpha=params["alpha"],
                    imbalance_penalty=params["imbalance_penalty"],
                    equalize_cluster_weights=self.equalize_cluster_weights,
                )
                forest_small.fit(
                    X=X,
                    y=y_residual,
                    sample_weight=sample_weight,
                    cluster=cluster,
                    compute_oob_predictions=True,
                )
                step_error = forest_small.grf_forest_["debiased_error"]
                if not np.nanmean(
                    step_error
                ) <= self.boost_error_reduction * np.nanmean(debiased_error):
                    break

            forest_residual = GRFForestRegressor(
                sample_fraction=params["sample_fraction"],
                mtry=params["mtry"],
                n_estimators=self.n_estimators,
                n_jobs=self.n_jobs,
                min_node_size=params["min_node_size"],
                honesty=self.honesty,
                honesty_fraction=params["honesty_fraction"],
                honesty_prune_leaves=params["honesty_prune_leaves"],
                seed=self.seed,
                ci_group_size=self.ci_group_size,
                alpha=params["alpha"],
                imbalance_penalty=params["imbalance_penalty"],
                equalize_cluster_weights=self.equalize_cluster_weights,
            )
            forest_residual.fit(
                X,
                y_residual,
                sample_weight=sample_weight,
                cluster=cluster,
                compute_oob_predictions=True,
            )
            current_pred = {
                "predictions": forest_residual.grf_forest_["predictions"],
                "debiased_error": forest_residual.grf_forest_["debiased_error"],
                "excess_error": forest_residual.grf_forest_["excess_error"],
            }
            y_hat = y_hat + np.atleast_1d(
                np.squeeze(np.array(current_pred["predictions"]))
            )
            debiased_error = current_pred["debiased_error"]
            boosted_forests["forest"].append(forest_residual)
            boosted_forests["error"].append(np.mean(debiased_error))
            step += 1
        # endregion

        boosted_forests["predictions"] = y_hat
        self.boosted_forests_ = boosted_forests
        return self

    def predict(self, X, boost_predict_steps=None):
        """Predict regression target for X.

        :param array2d X: prediction input features
        :param int boost_predict_steps: number of boost prediction steps
        """
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)
        num_forests = len(self.boosted_forests_["forest"])
        if boost_predict_steps is None:
            boost_predict_steps = num_forests
        else:
            boost_predict_steps = min(boost_predict_steps, num_forests)

        y_hat = 0
        for forest in self.boosted_forests_["forest"][:boost_predict_steps]:
            forest._ensure_ptr()
            result = grf.regression_predict(
                forest.grf_forest_cpp_,
                np.asfortranarray([[]]),  # train_matrix
                self.outcome_index_,
                np.asfortranarray(X.astype("float64")),  # test_matrix
                self._get_num_threads(),
                False,  # estimate_variance
            )
            y = np.atleast_1d(np.squeeze(np.array(result["predictions"])))
            y_hat = y_hat + y
        return y_hat

    def _more_tags(self):
        return {
            "requires_y": True,
            "_xfail_checks": {
                "check_sample_weights_invariance": "zero sample_weight is not equivalent to removing samples",
            },
        }


# region random parameter tuning functions
class GRFParamDistribution(ABC):
    def __init__(self, X_rows: int, X_cols: int):
        self.X_rows = X_rows
        self.X_cols = X_cols

    @abstractmethod
    def rvs(self, *args, **kwds):  # pragma: no cover
        raise NotImplementedError()


class GRFMinNodeSizeDistribution(GRFParamDistribution):
    def rvs(self, *args, **kwds):  # pragma: no cover
        return self.dist(ss.uniform(*args, **kwds))

    def dist(self, uniform):
        return np.floor(np.exp2(uniform * (np.log(self.X_rows) - np.log(2) - 4)))


class GRFSampleFractionDistribution(GRFParamDistribution):
    def rvs(self, *args, **kwds):  # pragma: no cover
        return self.dist(ss.uniform(*args, **kwds))

    def dist(self, uniform):
        return 0.05 + 0.45 * uniform


class GRFMtryDistribution(GRFParamDistribution):
    def rvs(self, *args, **kwds):  # pragma: no cover
        return self.dist(ss.uniform(*args, **kwds))

    def dist(self, uniform):
        return np.ceil(np.min([self.X_cols, np.sqrt(self.X_cols) + 20]) * uniform)


class GRFAlphaDistribution(GRFParamDistribution):
    def rvs(self, *args, **kwds):  # pragma: no cover
        return self.dist(ss.uniform(*args, **kwds))

    def dist(self, uniform):
        return uniform / 4


class GRFImbalancePenaltyDistribution(GRFParamDistribution):
    def rvs(self, *args, **kwds):  # pragma: no cover
        return self.dist(ss.uniform(*args, **kwds))

    def dist(self, uniform):
        return -np.log(uniform)


class GRFHonestyFractionDistribution(GRFParamDistribution):
    def rvs(self, *args, **kwds):  # pragma: no cover
        return self.dist(ss.uniform(*args, **kwds))

    def dist(self, uniform):
        return 0.5 + 0.3 * uniform


class GRFHonestyPruneLeavesDistribution(GRFParamDistribution):
    def __init__(self, X_rows: int, X_cols: int):
        super().__init__(X_rows, X_cols)
        self.choices = np.array([True, False])

    def rvs(self, *args, **kwds):  # pragma: no cover
        return self.dist(ss.uniform(*args, **kwds))

    def dist(self, uniform):
        return self.choices[(uniform < 0.5).astype(int)]


PARAM_DISTRIBUTIONS = {
    "min_node_size": GRFMinNodeSizeDistribution,
    "sample_fraction": GRFSampleFractionDistribution,
    "mtry": GRFMtryDistribution,
    "alpha": GRFAlphaDistribution,
    "imbalance_penalty": GRFImbalancePenaltyDistribution,
    "honesty_fraction": GRFHonestyFractionDistribution,
    "honesty_prune_leaves": GRFHonestyPruneLeavesDistribution,
}
# endregion
