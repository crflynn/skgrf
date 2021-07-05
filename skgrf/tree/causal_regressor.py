import typing as t

import logging

from skgrf.tree.instrumental_regressor import GRFTreeInstrumentalRegressor

if t.TYPE_CHECKING:  # pragma: no cover
    from skgrf.ensemble import GRFForestCausalRegressor

logger = logging.getLogger(__name__)


class GRFTreeCausalRegressor(GRFTreeInstrumentalRegressor):
    r"""GRF Tree Causal regression implementation for sci-kit learn.

    Provides a sklearn tree causal regressor to the GRF C++ library using Cython.

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
    :param bool orthogonal_boosting: When ``y_hat`` or ``w_hat`` are ``None``, they
        are estimated using boosted regression forests. (Not yet implemented)
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
    :ivar list clusters\_: The cluster labels determined from the fit input ``cluster``.
    :ivar int n_clusters\_: The number of unique cluster labels from the fit input
        ``cluster``.
    :ivar str criterion: The criterion used for splitting: ``mse``
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
        stabilize_splits=True,
        orthogonal_boosting=False,
        n_jobs=-1,
        seed=42,
    ):
        super().__init__(
            equalize_cluster_weights=equalize_cluster_weights,
            sample_fraction=sample_fraction,
            mtry=mtry,
            min_node_size=min_node_size,
            honesty=honesty,
            honesty_fraction=honesty_fraction,
            honesty_prune_leaves=honesty_prune_leaves,
            alpha=alpha,
            imbalance_penalty=imbalance_penalty,
            reduced_form_weight=0,
            stabilize_splits=stabilize_splits,
            n_jobs=n_jobs,
            seed=seed,
        )
        self.orthogonal_boosting = orthogonal_boosting

    @classmethod
    def from_forest(cls, forest: "GRFForestCausalRegressor", idx: int):
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
        grf_forest["num_trees"] = 1
        instance.grf_forest_ = grf_forest
        instance._ensure_ptr()
        # vars
        instance.outcome_index_ = forest.outcome_index_
        instance.treatment_index_ = forest.treatment_index_
        instance.instrument_index_ = forest.instrument_index_
        instance.n_features_in_ = forest.n_features_in_
        instance.clusters_ = forest.clusters_
        instance.n_clusters_ = forest.n_clusters_
        instance.samples_per_cluster_ = forest.samples_per_cluster_
        instance.mtry_ = forest.mtry_
        instance.sample_weight_index_ = forest.sample_weight_index_
        return instance

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
        # avoiding circular import
        from skgrf.ensemble.boosted_regressor import GRFBoostedForestRegressor

        X, y = self._validate_data(X, y)
        self._check_num_samples(X)

        boost_params = {
            "n_estimators": 500,
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
