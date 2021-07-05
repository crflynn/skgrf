import logging
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from skgrf.ensemble.boosted_regressor import GRFBoostedForestRegressor
from skgrf.ensemble.instrumental_regressor import GRFForestInstrumentalRegressor
from skgrf.tree.causal_regressor import GRFTreeCausalRegressor

logger = logging.getLogger(__name__)


class GRFForestCausalRegressor(GRFForestInstrumentalRegressor):
    r"""GRF Causal regression implementation for sci-kit learn.

    Provides a sklearn causal regressor to the GRF C++ library using Cython.

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
    :param bool orthogonal_boosting: When ``y_hat`` or ``w_hat`` are ``None``, they
        are estimated using boosted regression forests. (Not yet implemented)
    :param bool stabilize_splits: Whether or not the instrument should be taken into
        account when determining the imbalance of a split.
    :param int n_jobs: The number of threads. Default is number of CPU cores.
    :param int seed: Random seed value.

    :ivar list estimators\_: A list of tree objects from the forest.
    :ivar int n_features_in\_: The number of features (columns) from the fit input
        ``X``.
    :ivar dict grf_forest\_: The returned result object from calling C++ grf.
    :ivar int mtry\_: The ``mtry`` value determined by validation.
    :ivar int outcome_index\_: The index of the grf train matrix holding the outcomes.
    :ivar list samples_per_cluster\_: The number of samples to train per cluster.
    :ivar list clusters\_: The cluster labels determined from the fit input ``cluster``.
    :ivar int n_clusters\_: The number of unique cluster labels from the fit input
        ``cluster``.
    :ivar str criterion: The criterion used for splitting: ``mse``
    """

    def __init__(
        self,
        n_estimators=2000,
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
        stabilize_splits=True,
        orthogonal_boosting=False,
        n_jobs=-1,
        seed=42,
    ):
        super().__init__(
            n_estimators=n_estimators,
            equalize_cluster_weights=equalize_cluster_weights,
            sample_fraction=sample_fraction,
            mtry=mtry,
            min_node_size=min_node_size,
            honesty=honesty,
            honesty_fraction=honesty_fraction,
            honesty_prune_leaves=honesty_prune_leaves,
            alpha=alpha,
            imbalance_penalty=imbalance_penalty,
            ci_group_size=ci_group_size,
            reduced_form_weight=0,
            stabilize_splits=stabilize_splits,
            n_jobs=n_jobs,
            seed=seed,
        )
        self.orthogonal_boosting = orthogonal_boosting

    @property
    def estimators_(self):
        try:
            check_is_fitted(self)
        except NotFittedError:
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute 'estimators_'"
            ) from None
        return [
            GRFTreeCausalRegressor.from_forest(self, idx=idx)
            for idx in range(self.n_estimators)
        ]

    def get_estimator(self, idx):
        """Extract a single estimator tree from the forest.

        :param int idx: The index of the tree to extract.
        """
        check_is_fitted(self)
        return GRFTreeCausalRegressor.from_forest(self, idx=idx)

    # noinspection PyMethodOverriding
    def fit(
        self,
        X,
        y,
        w,  # treatment
        y_hat=None,
        w_hat=None,
        sample_weight=None,
        cluster=None,
    ):
        """Fit the grf forest using training data.

        :param array2d X: training input features
        :param array1d y: training input targets
        :param array1d w: training input treatments
        :param array1d y_hat: estimated expected target responses
        :param array1d w_hat: estimated treatment propensities
        :param array1d sample_weight: optional weights for input samples
        :param array1d cluster: optional cluster assignments for input samples
        """
        X, y = self._validate_data(X, y)
        self._check_num_samples(X)

        boost_params = {
            "n_estimators": max(50, int(self.n_estimators / 4)),
            "equalize_cluster_weights": self.equalize_cluster_weights,
            "sample_fraction": self.sample_fraction,
            "mtry": self.mtry,
            "min_node_size": 5,
            "honesty": True,
            "honesty_fraction": 0.5,
            "honesty_prune_leaves": self.honesty_prune_leaves,
            "alpha": self.alpha,
            "imbalance_penalty": self.imbalance_penalty,
            "ci_group_size": 1,
            "tune_params": None,  # TODO ?
            "n_jobs": self.n_jobs,
            "seed": self.seed,
        }
        if y_hat is None and self.orthogonal_boosting:
            logger.debug("orthogonal boosting y_hat")
            br = GRFBoostedForestRegressor(**boost_params)
            br.fit(X, y, sample_weight=sample_weight, cluster=cluster)
            y_hat = br.boosted_forests_["predictions"]

        if w_hat is None and self.orthogonal_boosting:
            logger.debug("orthogonal boosting w_hat")
            br = GRFBoostedForestRegressor(**boost_params)
            br.fit(X, w, sample_weight=sample_weight, cluster=cluster)
            w_hat = br.boosted_forests_["predictions"]

        return super().fit(
            X=X,
            y=y,
            w=w,
            z=w,
            y_hat=y_hat,
            w_hat=w_hat,
            z_hat=w_hat,
            cluster=cluster,
            sample_weight=sample_weight,
        )
