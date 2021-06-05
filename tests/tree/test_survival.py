import numpy as np
import pickle
import pytest
import tempfile
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.tree._tree import csr_matrix
from sklearn.utils.validation import check_is_fitted

from skgrf.ensemble.survival import GRFSurvival
from skgrf.tree.survival import GRFTreeSurvival


class TestGRFTreeSurvival:
    def test_init(self):
        _ = GRFTreeSurvival()

    def test_fit(self, lung_X, lung_y):
        tree = GRFTreeSurvival()
        with pytest.raises(NotFittedError):
            check_is_fitted(tree)
        tree.fit(lung_X, lung_y)
        check_is_fitted(tree)
        assert hasattr(tree, "grf_forest_")
        assert hasattr(tree, "mtry_")
        assert tree.grf_forest_["num_trees"] == 1

    def test_predict(self, lung_X, lung_y):
        tree = GRFTreeSurvival()
        tree.fit(lung_X, lung_y)
        pred = tree.predict(lung_X)
        assert len(pred) == lung_X.shape[0]

    def test_serialize(self, lung_X, lung_y):
        tree = GRFTreeSurvival()
        # not fitted
        tf = tempfile.TemporaryFile()
        pickle.dump(tree, tf)
        tf.seek(0)
        tree = pickle.load(tf)
        tree.fit(lung_X, lung_y)
        # fitted
        tf = tempfile.TemporaryFile()
        pickle.dump(tree, tf)
        tf.seek(0)
        new_tree = pickle.load(tf)
        pred = new_tree.predict(lung_X)
        assert len(pred) == lung_X.shape[0]

    def test_clone(self, lung_X, lung_y):
        tree = GRFTreeSurvival()
        tree.fit(lung_X, lung_y)
        clone(tree)

    def test_equalize_cluster_weights(
        self, lung_X, lung_y, lung_cluster, equalize_cluster_weights
    ):
        tree = GRFTreeSurvival(equalize_cluster_weights=equalize_cluster_weights)
        tree.fit(lung_X, lung_y, cluster=lung_cluster)
        if equalize_cluster_weights:
            assert tree.samples_per_cluster_ == 20
        else:
            assert tree.samples_per_cluster_ == lung_y.shape[0] - 20

        if equalize_cluster_weights:
            with pytest.raises(ValueError):
                tree.fit(
                    lung_X,
                    lung_y,
                    cluster=lung_cluster,
                    sample_weight=np.ones(lung_y.shape),
                )

        tree.fit(lung_X, lung_y, cluster=None)
        assert tree.samples_per_cluster_ == 0

    def test_sample_fraction(self, lung_X, lung_y, sample_fraction):
        tree = GRFTreeSurvival(sample_fraction=sample_fraction)
        if sample_fraction <= 0 or sample_fraction > 1:
            with pytest.raises(ValueError):
                tree.fit(lung_X, lung_y)
        else:
            tree.fit(lung_X, lung_y)

    def test_mtry(self, lung_X, lung_y, mtry):
        tree = GRFTreeSurvival(mtry=mtry)
        tree.fit(lung_X, lung_y)
        if mtry is not None:
            assert tree.mtry_ == mtry
        else:
            assert tree.mtry_ == 3

    def test_honesty(self, lung_X, lung_y, honesty):
        tree = GRFTreeSurvival(honesty=honesty)
        tree.fit(lung_X, lung_y)

    def test_honesty_fraction(self, lung_X, lung_y, honesty_fraction):
        tree = GRFTreeSurvival(
            honesty=True, honesty_fraction=honesty_fraction, honesty_prune_leaves=True
        )
        if honesty_fraction <= 0 or honesty_fraction >= 1:
            with pytest.raises(RuntimeError):
                tree.fit(lung_X, lung_y)
        else:
            tree.fit(lung_X, lung_y)

    def test_honesty_prune_leaves(self, lung_X, lung_y, honesty_prune_leaves):
        tree = GRFTreeSurvival(honesty=True, honesty_prune_leaves=honesty_prune_leaves)
        tree.fit(lung_X, lung_y)

    def test_alpha(self, lung_X, lung_y, alpha):
        tree = GRFTreeSurvival(alpha=alpha)
        if alpha <= 0 or alpha >= 0.25:
            with pytest.raises(ValueError):
                tree.fit(lung_X, lung_y)
        else:
            tree.fit(lung_X, lung_y)

    def test_get_tags(self):
        rfs = GRFTreeSurvival()
        tags = rfs._get_tags()
        assert tags["requires_y"]

    # cant use this because of special fit y
    # def test_check_estimator(self):
    #     check_estimator(GRFTreeSurvival())

    def test_from_forest(self, lung_X, lung_y):
        forest = GRFSurvival()
        forest.fit(lung_X, lung_y)
        tree = GRFTreeSurvival.from_forest(forest=forest, idx=0)
        tree.predict(lung_X)

    # region base
    def test_get_depth(self, lung_X, lung_y):
        tree = GRFTreeSurvival()
        tree.fit(lung_X, lung_y)
        depth = tree.get_depth()
        assert isinstance(depth, int)
        assert depth > 0

    def test_get_n_leaves(self, lung_X, lung_y):
        tree = GRFTreeSurvival()
        tree.fit(lung_X, lung_y)
        leaves = tree.get_n_leaves()
        assert isinstance(leaves, int)
        assert np.all(leaves > 0)

    def test_apply(self, lung_X, lung_y):
        tree = GRFTreeSurvival()
        tree.fit(lung_X, lung_y)
        leaves = tree.apply(lung_X)
        assert isinstance(leaves, np.ndarray)
        assert np.all(leaves > 0)
        assert len(leaves) == len(lung_X)

    def test_decision_path(self, lung_X, lung_y):
        tree = GRFTreeSurvival()
        tree.quantiles = [0.2]
        tree.fit(lung_X, lung_y)
        paths = tree.decision_path(lung_X)
        assert isinstance(paths, csr_matrix)
        assert paths.shape[0] == len(lung_X)

    # endregion
