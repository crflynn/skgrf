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
        gfi = GRFTreeInstrumentalRegressor()
        with pytest.raises(NotFittedError):
            check_is_fitted(gfi)
        gfi.fit(causal_X, causal_y, causal_w, causal_w)
        check_is_fitted(gfi)
        assert hasattr(gfi, "grf_forest_")
        assert hasattr(gfi, "mtry_")

    def test_predict(self, causal_X, causal_y, causal_w):
        gfi = GRFTreeInstrumentalRegressor()
        gfi.fit(causal_X, causal_y, causal_w, causal_w)
        pred = gfi.predict(causal_X)
        assert len(pred) == causal_X.shape[0]

    def test_serialize(self, causal_X, causal_y, causal_w):
        gfi = GRFTreeInstrumentalRegressor()
        # not fitted
        tf = tempfile.TemporaryFile()
        pickle.dump(gfi, tf)
        tf.seek(0)
        gfi = pickle.load(tf)
        gfi.fit(causal_X, causal_y, causal_w, causal_w)
        # fitted
        tf = tempfile.TemporaryFile()
        pickle.dump(gfi, tf)
        tf.seek(0)
        new_gfi = pickle.load(tf)
        pred = new_gfi.predict(causal_X)
        assert len(pred) == causal_X.shape[0]

    def test_clone(self, causal_X, causal_y, causal_w):
        gfi = GRFTreeInstrumentalRegressor()
        gfi.fit(causal_X, causal_y, causal_w, causal_w)
        clone(gfi)

    def test_equalize_cluster_weights(
        self,
        causal_X,
        causal_y,
        causal_w,
        causal_cluster,
        equalize_cluster_weights,
    ):
        gfi = GRFTreeInstrumentalRegressor(
            equalize_cluster_weights=equalize_cluster_weights
        )
        gfi.fit(causal_X, causal_y, causal_w, causal_w, cluster=causal_cluster)
        if equalize_cluster_weights:
            assert gfi.samples_per_cluster_ == 20
        else:
            assert gfi.samples_per_cluster_ == causal_y.shape[0] - 20

        if equalize_cluster_weights:
            with pytest.raises(ValueError):
                gfi.fit(
                    causal_X,
                    causal_y,
                    causal_w,
                    causal_w,
                    cluster=causal_cluster,
                    sample_weight=causal_y,
                )

        gfi.fit(causal_X, causal_y, causal_w, causal_w, cluster=None)
        assert gfi.samples_per_cluster_ == 0

    def test_sample_fraction(self, causal_X, causal_y, causal_w, sample_fraction):
        gfi = GRFTreeInstrumentalRegressor(sample_fraction=sample_fraction)
        if sample_fraction <= 0 or sample_fraction > 1:
            with pytest.raises(ValueError):
                gfi.fit(causal_X, causal_y, causal_w, causal_w)
        else:
            gfi.fit(causal_X, causal_y, causal_w, causal_w)

    def test_mtry(self, causal_X, causal_y, causal_w, mtry):
        gfi = GRFTreeInstrumentalRegressor(mtry=mtry)
        gfi.fit(causal_X, causal_y, causal_w, causal_w)
        if mtry is not None:
            assert gfi.mtry_ == mtry
        else:
            assert gfi.mtry_ == 5

    def test_honesty(self, causal_X, causal_y, causal_w, honesty):
        gfi = GRFTreeInstrumentalRegressor(honesty=honesty)
        gfi.fit(causal_X, causal_y, causal_w, causal_w)

    def test_honesty_fraction(self, causal_X, causal_y, causal_w, honesty_fraction):
        gfi = GRFTreeInstrumentalRegressor(
            honesty=True, honesty_fraction=honesty_fraction, honesty_prune_leaves=True
        )
        if honesty_fraction <= 0 or honesty_fraction >= 1:
            with pytest.raises(RuntimeError):
                gfi.fit(causal_X, causal_y, causal_w, causal_w)
        else:
            gfi.fit(causal_X, causal_y, causal_w, causal_w)

    def test_honesty_prune_leaves(
        self, causal_X, causal_y, causal_w, honesty_prune_leaves
    ):
        gfi = GRFTreeInstrumentalRegressor(
            honesty=True, honesty_prune_leaves=honesty_prune_leaves
        )
        gfi.fit(causal_X, causal_y, causal_w, causal_w)

    def test_alpha(self, causal_X, causal_y, causal_w, alpha):
        gfi = GRFTreeInstrumentalRegressor(alpha=alpha)
        if alpha <= 0 or alpha >= 0.25:
            with pytest.raises(ValueError):
                gfi.fit(causal_X, causal_y, causal_w, causal_w)
        else:
            gfi.fit(causal_X, causal_y, causal_w, causal_w)

    # cant use this because of extra required fit params
    # def test_check_estimator(self):
    #     check_estimator(GRFTreeInstrumentalRegressor())

    def test_from_forest(self, causal_X, causal_y, causal_w):
        gfi = GRFInstrumentalRegressor()
        gfi.fit(causal_X, causal_y, causal_w, causal_w)
        tree = GRFTreeInstrumentalRegressor.from_forest(forest=gfi, idx=0)
        tree.predict(causal_X)

    # region base
    def test_get_depth(self, causal_X, causal_y, causal_w):
        gfi = GRFTreeInstrumentalRegressor()
        gfi.fit(causal_X, causal_y, causal_w, causal_w)
        assert gfi.get_depth() == 8

    def test_get_n_leaves(self, causal_X, causal_y, causal_w):
        gfi = GRFTreeInstrumentalRegressor()
        gfi.fit(causal_X, causal_y, causal_w, causal_w)
        assert gfi.get_n_leaves() == 12

    def test_apply(self, causal_X, causal_y, causal_w):
        gfi = GRFTreeInstrumentalRegressor()
        gfi.fit(causal_X, causal_y, causal_w, causal_w)
        np.testing.assert_equal(gfi.apply(causal_X[:3, :]), [9, 9, 9])

    def test_decision_path(self, causal_X, causal_y, causal_w):
        gfi = GRFTreeInstrumentalRegressor()
        gfi.fit(causal_X, causal_y, causal_w, causal_w)
        paths = gfi.decision_path(causal_X)
        assert isinstance(paths, csr_matrix)

    # endregion
