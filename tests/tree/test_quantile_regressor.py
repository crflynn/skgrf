import pickle
import pytest
import tempfile
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted

from skgrf.ensemble import GRFQuantileRegressor
from skgrf.tree import GRFTreeQuantileRegressor


class TestGRFTreeQuantileRegressor:
    def test_init(self):
        _ = GRFTreeQuantileRegressor()

    def test_fit(self, boston_X, boston_y):
        gqr = GRFTreeQuantileRegressor()
        with pytest.raises(NotFittedError):
            check_is_fitted(gqr)
        with pytest.raises(ValueError):
            gqr.fit(boston_X, boston_y)
        gqr.quantiles = [0.2, 0.5, 0.8]
        gqr.fit(boston_X, boston_y)
        check_is_fitted(gqr)
        assert hasattr(gqr, "grf_forest_")
        assert hasattr(gqr, "mtry_")
        assert gqr.grf_forest_["_num_trees"] == 1

    def test_predict(self, boston_X, boston_y):
        gqr = GRFTreeQuantileRegressor()
        gqr.quantiles = [0.2, 0.5, 0.8]
        gqr.fit(boston_X, boston_y)
        pred = gqr.predict(boston_X)
        assert len(pred) == boston_X.shape[0]

    def test_serialize(self, boston_X, boston_y):
        gqr = GRFTreeQuantileRegressor()
        gqr.quantiles = [0.2, 0.5, 0.8]
        # not fitted
        tf = tempfile.TemporaryFile()
        pickle.dump(gqr, tf)
        tf.seek(0)
        gqr = pickle.load(tf)
        gqr.fit(boston_X, boston_y)
        # fitted
        tf = tempfile.TemporaryFile()
        pickle.dump(gqr, tf)
        tf.seek(0)
        new_gqr = pickle.load(tf)
        pred = new_gqr.predict(boston_X)
        assert len(pred) == boston_X.shape[0]

    def test_clone(self, boston_X, boston_y):
        gqr = GRFTreeQuantileRegressor()
        gqr.quantiles = [0.2, 0.5, 0.8]
        gqr.fit(boston_X, boston_y)
        clone(gqr)

    def test_equalize_cluster_weights(
        self, boston_X, boston_y, boston_cluster, equalize_cluster_weights
    ):
        gqr = GRFTreeQuantileRegressor(
            equalize_cluster_weights=equalize_cluster_weights
        )
        gqr.quantiles = [0.2, 0.5, 0.8]
        gqr.fit(boston_X, boston_y, cluster=boston_cluster)
        if equalize_cluster_weights:
            assert gqr.samples_per_cluster_ == 20
        else:
            assert gqr.samples_per_cluster_ == boston_y.shape[0] - 20

        gqr.fit(boston_X, boston_y, cluster=boston_cluster)
        gqr.fit(boston_X, boston_y, cluster=None)
        assert gqr.samples_per_cluster_ == 0

    def test_sample_fraction(self, boston_X, boston_y, sample_fraction):
        gqr = GRFTreeQuantileRegressor(sample_fraction=sample_fraction)
        gqr.quantiles = [0.2, 0.5, 0.8]
        if sample_fraction <= 0 or sample_fraction > 1:
            with pytest.raises(ValueError):
                gqr.fit(boston_X, boston_y)
        else:
            gqr.fit(boston_X, boston_y)

    def test_mtry(self, boston_X, boston_y, mtry):
        gqr = GRFTreeQuantileRegressor(mtry=mtry)
        gqr.quantiles = [0.2, 0.5, 0.8]
        gqr.fit(boston_X, boston_y)
        if mtry is not None:
            assert gqr.mtry_ == mtry
        else:
            assert gqr.mtry_ == 6

    def test_honesty(self, boston_X, boston_y, honesty):
        gqr = GRFTreeQuantileRegressor(honesty=honesty)
        gqr.quantiles = [0.2, 0.5, 0.8]
        gqr.fit(boston_X, boston_y)

    def test_honesty_fraction(self, boston_X, boston_y, honesty_fraction):
        gqr = GRFTreeQuantileRegressor(
            honesty=True, honesty_fraction=honesty_fraction, honesty_prune_leaves=True
        )
        gqr.quantiles = [0.2, 0.5, 0.8]
        if honesty_fraction <= 0 or honesty_fraction >= 1:
            with pytest.raises(RuntimeError):
                gqr.fit(boston_X, boston_y)
        else:
            gqr.fit(boston_X, boston_y)

    def test_honesty_prune_leaves(self, boston_X, boston_y, honesty_prune_leaves):
        gqr = GRFTreeQuantileRegressor(
            honesty=True, honesty_prune_leaves=honesty_prune_leaves
        )
        gqr.quantiles = [0.2, 0.5, 0.8]
        gqr.fit(boston_X, boston_y)

    def test_alpha(self, boston_X, boston_y, alpha):
        gqr = GRFTreeQuantileRegressor(alpha=alpha)
        gqr.quantiles = [0.2, 0.5, 0.8]
        if alpha <= 0 or alpha >= 0.25:
            with pytest.raises(ValueError):
                gqr.fit(boston_X, boston_y)
        else:
            gqr.fit(boston_X, boston_y)

    def test_check_estimator(self):
        check_estimator(GRFTreeQuantileRegressor(quantiles=[0.2]))

    def test_from_forest(self, boston_X, boston_y):
        gqr = GRFQuantileRegressor(quantiles=[0.2])
        gqr.fit(boston_X, boston_y)
        tree = GRFTreeQuantileRegressor.from_forest(forest=gqr, idx=0)
        tree.predict(boston_X)
