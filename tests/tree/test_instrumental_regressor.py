import numpy as np
import pickle
import pytest
import tempfile
from sklearn import clone
from sklearn.exceptions import NotFittedError
from sklearn.tree._tree import csr_matrix
from sklearn.utils.validation import check_is_fitted

from skgrf.ensemble import GRFInstrumentalRegressor
from skgrf.tree import GRFTreeInstrumentalRegressor


class TestGRFTreeInstrumentalRegressor:
    def test_init(self):
        _ = GRFTreeInstrumentalRegressor()

    def test_fit(self, causal_X, causal_y, causal_w):
        tree = GRFTreeInstrumentalRegressor()
        with pytest.raises(NotFittedError):
            check_is_fitted(tree)
        tree.fit(causal_X, causal_y, causal_w, causal_w)
        check_is_fitted(tree)
        assert hasattr(tree, "grf_forest_")
        assert hasattr(tree, "mtry_")
        assert tree.grf_forest_["num_trees"] == 1

    def test_predict(self, causal_X, causal_y, causal_w):
        tree = GRFTreeInstrumentalRegressor()
        tree.fit(causal_X, causal_y, causal_w, causal_w)
        pred = tree.predict(causal_X)
        assert len(pred) == causal_X.shape[0]

    def test_serialize(self, causal_X, causal_y, causal_w):
        tree = GRFTreeInstrumentalRegressor()
        # not fitted
        tf = tempfile.TemporaryFile()
        pickle.dump(tree, tf)
        tf.seek(0)
        tree = pickle.load(tf)
        tree.fit(causal_X, causal_y, causal_w, causal_w)
        # fitted
        tf = tempfile.TemporaryFile()
        pickle.dump(tree, tf)
        tf.seek(0)
        new_tree = pickle.load(tf)
        pred = new_tree.predict(causal_X)
        assert len(pred) == causal_X.shape[0]

    def test_clone(self, causal_X, causal_y, causal_w):
        tree = GRFTreeInstrumentalRegressor()
        tree.fit(causal_X, causal_y, causal_w, causal_w)
        clone(tree)

    def test_equalize_cluster_weights(
        self,
        causal_X,
        causal_y,
        causal_w,
        causal_cluster,
        equalize_cluster_weights,
    ):
        tree = GRFTreeInstrumentalRegressor(
            equalize_cluster_weights=equalize_cluster_weights
        )
        tree.fit(causal_X, causal_y, causal_w, causal_w, cluster=causal_cluster)
        if equalize_cluster_weights:
            assert tree.samples_per_cluster_ == 20
        else:
            assert tree.samples_per_cluster_ == causal_y.shape[0] - 20

        if equalize_cluster_weights:
            with pytest.raises(ValueError):
                tree.fit(
                    causal_X,
                    causal_y,
                    causal_w,
                    causal_w,
                    cluster=causal_cluster,
                    sample_weight=causal_y,
                )

        tree.fit(causal_X, causal_y, causal_w, causal_w, cluster=None)
        assert tree.samples_per_cluster_ == 0

    def test_sample_fraction(self, causal_X, causal_y, causal_w, sample_fraction):
        tree = GRFTreeInstrumentalRegressor(sample_fraction=sample_fraction)
        if sample_fraction <= 0 or sample_fraction > 1:
            with pytest.raises(ValueError):
                tree.fit(causal_X, causal_y, causal_w, causal_w)
        else:
            tree.fit(causal_X, causal_y, causal_w, causal_w)

    def test_mtry(self, causal_X, causal_y, causal_w, mtry):
        tree = GRFTreeInstrumentalRegressor(mtry=mtry)
        tree.fit(causal_X, causal_y, causal_w, causal_w)
        if mtry is not None:
            assert tree.mtry_ == mtry
        else:
            assert tree.mtry_ == 5

    def test_honesty(self, causal_X, causal_y, causal_w, honesty):
        tree = GRFTreeInstrumentalRegressor(honesty=honesty)
        tree.fit(causal_X, causal_y, causal_w, causal_w)

    def test_honesty_fraction(self, causal_X, causal_y, causal_w, honesty_fraction):
        tree = GRFTreeInstrumentalRegressor(
            honesty=True, honesty_fraction=honesty_fraction, honesty_prune_leaves=True
        )
        if honesty_fraction <= 0 or honesty_fraction >= 1:
            with pytest.raises(RuntimeError):
                tree.fit(causal_X, causal_y, causal_w, causal_w)
        else:
            tree.fit(causal_X, causal_y, causal_w, causal_w)

    def test_honesty_prune_leaves(
        self, causal_X, causal_y, causal_w, honesty_prune_leaves
    ):
        tree = GRFTreeInstrumentalRegressor(
            honesty=True, honesty_prune_leaves=honesty_prune_leaves
        )
        tree.fit(causal_X, causal_y, causal_w, causal_w)

    def test_alpha(self, causal_X, causal_y, causal_w, alpha):
        tree = GRFTreeInstrumentalRegressor(alpha=alpha)
        if alpha <= 0 or alpha >= 0.25:
            with pytest.raises(ValueError):
                tree.fit(causal_X, causal_y, causal_w, causal_w)
        else:
            tree.fit(causal_X, causal_y, causal_w, causal_w)

    # cant use this because of extra required fit params
    # def test_check_estimator(self):
    #     check_estimator(GRFTreeInstrumentalRegressor())

    def test_from_forest(self, causal_X, causal_y, causal_w):
        forest = GRFInstrumentalRegressor()
        forest.fit(causal_X, causal_y, causal_w, causal_w)
        tree = GRFTreeInstrumentalRegressor.from_forest(forest=forest, idx=0)
        tree.predict(causal_X)

    # region base
    def test_get_depth(self, causal_X, causal_y, causal_w):
        tree = GRFTreeInstrumentalRegressor()
        tree.fit(causal_X, causal_y, causal_w, causal_w)
        depth = tree.get_depth()
        assert isinstance(depth, int)
        assert depth > 0

    def test_get_n_leaves(self, causal_X, causal_y, causal_w):
        tree = GRFTreeInstrumentalRegressor()
        tree.fit(causal_X, causal_y, causal_w, causal_w)
        leaves = tree.get_n_leaves()
        assert isinstance(leaves, int)
        assert np.all(leaves > 0)

    def test_apply(self, causal_X, causal_y, causal_w):
        tree = GRFTreeInstrumentalRegressor()
        tree.fit(causal_X, causal_y, causal_w, causal_w)
        leaves = tree.apply(causal_X)
        assert isinstance(leaves, np.ndarray)
        assert np.all(leaves > 0)
        assert len(leaves) == len(causal_X)

    def test_decision_path(self, causal_X, causal_y, causal_w):
        tree = GRFTreeInstrumentalRegressor()
        tree.fit(causal_X, causal_y, causal_w, causal_w)
        paths = tree.decision_path(causal_X)
        assert isinstance(paths, csr_matrix)
        assert paths.shape[0] == len(causal_X)

    # endregion
