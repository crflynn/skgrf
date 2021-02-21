import numpy as np
import pickle
import pytest
import tempfile
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from skgrf.ensemble import GRFRegressor


class TestGRFRegressor:
    def test_init(self):
        _ = GRFRegressor()

    def test_fit(self, boston_X, boston_y):
        gfr = GRFRegressor()
        with pytest.raises(NotFittedError):
            check_is_fitted(gfr)
        gfr.fit(boston_X, boston_y)
        check_is_fitted(gfr)
        assert hasattr(gfr, "grf_forest_")
        assert hasattr(gfr, "mtry_")

    def test_predict(self, boston_X, boston_y):
        gfr = GRFRegressor()
        gfr.fit(boston_X, boston_y)
        pred = gfr.predict(boston_X)
        print(pred)
        print(boston_X.shape)
        assert len(pred) == boston_X.shape[0]

    def test_predictoob(self, boston_X, boston_y):
        gfr = GRFRegressor()
        gfr.fit(boston_X, boston_y, compute_oob_predictions=True)
        pred = np.atleast_1d(np.squeeze(np.array(gfr.grf_forest_["predictions"])))
        assert len(pred) == boston_X.shape[0]

    def test_serialize(self, boston_X, boston_y):
        tf = tempfile.TemporaryFile()
        gfr = GRFRegressor()
        gfr.fit(boston_X, boston_y)
        pickle.dump(gfr, tf)
        tf.seek(0)
        new_gfr = pickle.load(tf)
        pred = new_gfr.predict(boston_X)
        assert len(pred) == boston_X.shape[0]

    def test_clone(self, boston_X, boston_y):
        gfr = GRFRegressor()
        gfr.fit(boston_X, boston_y)
        clone(gfr)

    def test_equalize_cluster_weights(
        self, boston_X, boston_y, boston_cluster, equalize_cluster_weights
    ):
        gfr = GRFRegressor(equalize_cluster_weights=equalize_cluster_weights)
        gfr.fit(boston_X, boston_y, cluster=boston_cluster)
        if equalize_cluster_weights:
            assert gfr.samples_per_cluster_ == 20
        else:
            assert gfr.samples_per_cluster_ == boston_y.shape[0] - 20

        if equalize_cluster_weights:
            with pytest.raises(ValueError):
                gfr.fit(
                    boston_X, boston_y, cluster=boston_cluster, sample_weight=boston_y
                )

        gfr.fit(boston_X, boston_y, cluster=None)
        assert gfr.samples_per_cluster_ == 0

    def test_sample_fraction(
        self, boston_X, boston_y, sample_fraction
    ):  # and ci_group_size
        gfr = GRFRegressor(sample_fraction=sample_fraction, ci_group_size=1)
        if sample_fraction <= 0 or sample_fraction > 1:
            with pytest.raises(ValueError):
                gfr.fit(boston_X, boston_y)
        else:
            gfr.fit(boston_X, boston_y)

        gfr = GRFRegressor(sample_fraction=sample_fraction, ci_group_size=2)
        if sample_fraction <= 0 or sample_fraction > 0.5:
            with pytest.raises(ValueError):
                gfr.fit(boston_X, boston_y)
        else:
            gfr.fit(boston_X, boston_y)

    def test_mtry(self, boston_X, boston_y, mtry):
        gfr = GRFRegressor(mtry=mtry)
        gfr.fit(boston_X, boston_y)
        if mtry is not None:
            assert gfr.mtry_ == mtry
        else:
            assert gfr.mtry_ == 6

    def test_honesty(self, boston_X, boston_y, honesty):
        gfr = GRFRegressor(honesty=honesty)
        gfr.fit(boston_X, boston_y)

    def test_honesty_fraction(self, boston_X, boston_y, honesty_fraction):
        gfr = GRFRegressor(
            honesty=True, honesty_fraction=honesty_fraction, honesty_prune_leaves=True
        )
        if honesty_fraction <= 0 or honesty_fraction >= 1:
            with pytest.raises(RuntimeError):
                gfr.fit(boston_X, boston_y)
        else:
            gfr.fit(boston_X, boston_y)

    def test_honesty_prune_leaves(self, boston_X, boston_y, honesty_prune_leaves):
        gfr = GRFRegressor(honesty=True, honesty_prune_leaves=honesty_prune_leaves)
        gfr.fit(boston_X, boston_y)

    def test_alpha(self, boston_X, boston_y, alpha):
        gfr = GRFRegressor(alpha=alpha)
        if alpha <= 0 or alpha >= 0.25:
            with pytest.raises(ValueError):
                gfr.fit(boston_X, boston_y)
        else:
            gfr.fit(boston_X, boston_y)
