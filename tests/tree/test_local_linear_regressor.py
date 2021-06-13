import numpy as np
import pickle
import pytest
import tempfile
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.tree._tree import csr_matrix
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted

from skgrf.ensemble import GRFLocalLinearRegressor
from skgrf.tree import GRFTreeLocalLinearRegressor


class TestGRFTreeLocalLinearRegressor:
    def test_init(self):
        _ = GRFTreeLocalLinearRegressor()

    def test_fit(self, boston_X, boston_y):
        tree = GRFTreeLocalLinearRegressor(ll_split_cutoff=0)
        with pytest.raises(NotFittedError):
            check_is_fitted(tree)
        tree.fit(boston_X, boston_y)
        check_is_fitted(tree)
        assert hasattr(tree, "grf_forest_")
        assert hasattr(tree, "mtry_")
        assert tree.grf_forest_["num_trees"] == 1

    def test_predict(self, boston_X, boston_y):
        tree = GRFTreeLocalLinearRegressor(ll_split_cutoff=0)
        tree.fit(boston_X, boston_y)
        pred = tree.predict(boston_X)
        assert len(pred) == boston_X.shape[0]

    def test_serialize(self, boston_X, boston_y):
        tree = GRFTreeLocalLinearRegressor(ll_split_cutoff=0)
        # not fitted
        tf = tempfile.TemporaryFile()
        pickle.dump(tree, tf)
        tf.seek(0)
        tree = pickle.load(tf)
        tree.fit(boston_X, boston_y)
        # fitted
        tf = tempfile.TemporaryFile()
        pickle.dump(tree, tf)
        tf.seek(0)
        new_tree = pickle.load(tf)
        pred = new_tree.predict(boston_X)
        assert len(pred) == boston_X.shape[0]

    def test_clone(self, boston_X, boston_y):
        tree = GRFTreeLocalLinearRegressor(ll_split_cutoff=0)
        tree.fit(boston_X, boston_y)
        clone(tree)

    def test_equalize_cluster_weights(
        self, boston_X, boston_y, boston_cluster, equalize_cluster_weights
    ):
        tree = GRFTreeLocalLinearRegressor(
            equalize_cluster_weights=equalize_cluster_weights
        )
        tree.fit(boston_X, boston_y, cluster=boston_cluster)
        if equalize_cluster_weights:
            assert tree.samples_per_cluster_ == 20
        else:
            assert tree.samples_per_cluster_ == boston_y.shape[0] - 20

        if equalize_cluster_weights:
            with pytest.raises(ValueError):
                tree.fit(
                    boston_X, boston_y, cluster=boston_cluster, sample_weight=boston_y
                )

        tree.fit(boston_X, boston_y, cluster=None)
        assert tree.samples_per_cluster_ == 0

    def test_sample_fraction(
        self, boston_X, boston_y, sample_fraction
    ):  # and ci_group_size
        tree = GRFTreeLocalLinearRegressor(sample_fraction=sample_fraction)
        if sample_fraction <= 0 or sample_fraction > 1:
            with pytest.raises(ValueError):
                tree.fit(boston_X, boston_y)
        else:
            tree.fit(boston_X, boston_y)

    def test_mtry(self, boston_X, boston_y, mtry):
        tree = GRFTreeLocalLinearRegressor(mtry=mtry)
        tree.fit(boston_X, boston_y)
        if mtry is not None:
            assert tree.mtry_ == mtry
        else:
            assert tree.mtry_ == 6

    def test_honesty(self, boston_X, boston_y, honesty):
        tree = GRFTreeLocalLinearRegressor(honesty=honesty)
        tree.fit(boston_X, boston_y)

    def test_honesty_fraction(self, boston_X, boston_y, honesty_fraction):
        tree = GRFTreeLocalLinearRegressor(
            honesty=True, honesty_fraction=honesty_fraction, honesty_prune_leaves=True
        )
        if honesty_fraction <= 0 or honesty_fraction >= 1:
            with pytest.raises(RuntimeError):
                tree.fit(boston_X, boston_y)
        else:
            tree.fit(boston_X, boston_y)

    def test_honesty_prune_leaves(self, boston_X, boston_y, honesty_prune_leaves):
        tree = GRFTreeLocalLinearRegressor(
            honesty=True, honesty_prune_leaves=honesty_prune_leaves
        )
        tree.fit(boston_X, boston_y)

    def test_alpha(self, boston_X, boston_y, alpha):
        tree = GRFTreeLocalLinearRegressor(alpha=alpha)
        if alpha <= 0 or alpha >= 0.25:
            with pytest.raises(ValueError):
                tree.fit(boston_X, boston_y)
        else:
            tree.fit(boston_X, boston_y)

    def test_check_estimator(self):
        check_estimator(GRFTreeLocalLinearRegressor())

    def test_from_forest(self, boston_X, boston_y):
        forest = GRFLocalLinearRegressor()
        forest.fit(boston_X, boston_y)
        tree = GRFTreeLocalLinearRegressor.from_forest(forest=forest, idx=0)
        tree.predict(boston_X)

    # region base
    def test_get_depth(self, boston_X, boston_y):
        tree = GRFTreeLocalLinearRegressor()
        tree.fit(boston_X, boston_y)
        depth = tree.get_depth()
        assert isinstance(depth, int)
        assert depth > 0

    def test_get_n_leaves(self, boston_X, boston_y):
        tree = GRFTreeLocalLinearRegressor()
        tree.fit(boston_X, boston_y)
        leaves = tree.get_n_leaves()
        assert isinstance(leaves, int)
        assert np.all(leaves > 0)

    def test_apply(self, boston_X, boston_y):
        tree = GRFTreeLocalLinearRegressor()
        tree.fit(boston_X, boston_y)
        leaves = tree.apply(boston_X)
        assert isinstance(leaves, np.ndarray)
        assert np.all(leaves > 0)
        assert len(leaves) == len(boston_X)

    def test_decision_path(self, boston_X, boston_y):
        tree = GRFTreeLocalLinearRegressor()
        tree.fit(boston_X, boston_y)
        paths = tree.decision_path(boston_X)
        assert isinstance(paths, csr_matrix)
        assert paths.shape[0] == len(boston_X)

    def test_tree_interface(self, boston_X, boston_y):
        tree = GRFTreeLocalLinearRegressor()
        tree.fit(boston_X, boston_y)
        # access attributes the way we would expect to in sklearn
        tree_ = tree.tree_
        children_left = tree_.children_left
        children_right = tree_.children_right
        children_default = tree_.children_default
        feature = tree_.feature
        threshold = tree_.threshold
        max_depth = tree_.max_depth
        n_node_samples = tree_.n_node_samples
        weighted_n_node_samples = tree_.weighted_n_node_samples

    # endregion
