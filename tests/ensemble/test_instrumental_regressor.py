import pickle
import pytest
import tempfile
from sklearn import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from skgrf.ensemble import GRFInstrumentalRegressor


class TestGRFInstrumental:
    def test_init(self):
        _ = GRFInstrumentalRegressor()

    def test_fit(self, boston_X, boston_y):
        gfi = GRFInstrumentalRegressor()
        with pytest.raises(NotFittedError):
            check_is_fitted(gfi)
        gfi.fit(boston_X, boston_y, boston_y, boston_y)
        check_is_fitted(gfi)
        assert hasattr(gfi, "grf_forest_")
        assert hasattr(gfi, "mtry_")

    def test_predict(self, boston_X, boston_y):
        gfi = GRFInstrumentalRegressor()
        gfi.fit(boston_X, boston_y, boston_y, boston_y)
        pred = gfi.predict(boston_X)
        assert len(pred) == boston_X.shape[0]

    def test_serialize(self, boston_X, boston_y):
        tf = tempfile.TemporaryFile()
        gfi = GRFInstrumentalRegressor()
        gfi.fit(boston_X, boston_y, boston_y, boston_y)
        pickle.dump(gfi, tf)
        tf.seek(0)
        new_gfi = pickle.load(tf)
        pred = new_gfi.predict(boston_X)
        assert len(pred) == boston_X.shape[0]

    def test_clone(self, boston_X, boston_y):
        gfi = GRFInstrumentalRegressor()
        gfi.fit(boston_X, boston_y, boston_y, boston_y)
        clone(gfi)


    def test_equalize_cluster_weights(
        self, boston_X, boston_y, boston_cluster, equalize_cluster_weights
    ):
        gfi = GRFInstrumentalRegressor(equalize_cluster_weights=equalize_cluster_weights)
        gfi.fit(boston_X, boston_y, cluster=boston_cluster)
        if equalize_cluster_weights:
            assert gfi.samples_per_cluster_ == 20
        else:
            assert gfi.samples_per_cluster_ == boston_y.shape[0] - 20

        if equalize_cluster_weights:
            with pytest.raises(ValueError):
                gfi.fit(
                    boston_X, boston_y, cluster=boston_cluster, sample_weight=boston_y
                )

        gfi.fit(boston_X, boston_y, cluster=None)
        assert gfi.samples_per_cluster_ == 0

    def test_sample_fraction(
        self, boston_X, boston_y, sample_fraction
    ):  # and ci_group_size
        gfi = GRFInstrumentalRegressor(sample_fraction=sample_fraction, ci_group_size=1)
        if sample_fraction <= 0 or sample_fraction > 1:
            with pytest.raises(ValueError):
                gfi.fit(boston_X, boston_y)
        else:
            gfi.fit(boston_X, boston_y)

        gfi = GRFInstrumentalRegressor(sample_fraction=sample_fraction, ci_group_size=2)
        if sample_fraction <= 0 or sample_fraction > 0.5:
            with pytest.raises(ValueError):
                gfi.fit(boston_X, boston_y)
        else:
            gfi.fit(boston_X, boston_y)

    def test_mtry(self, boston_X, boston_y, mtry):
        gfi = GRFInstrumentalRegressor(mtry=mtry)
        gfi.fit(boston_X, boston_y)
        if mtry is not None:
            assert gfi.mtry_ == mtry
        else:
            assert gfi.mtry_ == 6

    def test_honesty(self, boston_X, boston_y, honesty):
        gfi = GRFInstrumentalRegressor(honesty=honesty)
        gfi.fit(boston_X, boston_y)

    def test_honesty_fraction(self, boston_X, boston_y, honesty_fraction):
        gfi = GRFInstrumentalRegressor(
            honesty=True, honesty_fraction=honesty_fraction, honesty_prune_leaves=True
        )
        if honesty_fraction <= 0 or honesty_fraction >= 1:
            with pytest.raises(RuntimeError):
                gfi.fit(boston_X, boston_y)
        else:
            gfi.fit(boston_X, boston_y)

    def test_honesty_prune_leaves(self, boston_X, boston_y, honesty_prune_leaves):
        gfi = GRFInstrumentalRegressor(honesty=True, honesty_prune_leaves=honesty_prune_leaves)
        gfi.fit(boston_X, boston_y)

    def test_alpha(self, boston_X, boston_y, alpha):
        gfi = GRFInstrumentalRegressor(alpha=alpha)
        if alpha <= 0 or alpha >= 0.25:
            with pytest.raises(ValueError):
                gfi.fit(boston_X, boston_y)
        else:
            gfi.fit(boston_X, boston_y)
