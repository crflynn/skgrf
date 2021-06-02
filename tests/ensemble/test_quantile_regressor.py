import pickle
import pytest
import tempfile
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted

from skgrf.ensemble import GRFQuantileRegressor
from skgrf.tree import GRFTreeQuantileRegressor


class TestGRFQuantileRegressor:
    def test_init(self):
        _ = GRFQuantileRegressor()

    def test_fit(self, boston_X, boston_y):
        forest = GRFQuantileRegressor()
        with pytest.raises(NotFittedError):
            check_is_fitted(forest)
        with pytest.raises(ValueError):
            forest.fit(boston_X, boston_y)
        forest.quantiles = [0.2, 0.5, 0.8]
        forest.fit(boston_X, boston_y)
        check_is_fitted(forest)
        assert hasattr(forest, "grf_forest_")
        assert hasattr(forest, "mtry_")

    def test_predict(self, boston_X, boston_y):
        forest = GRFQuantileRegressor()
        forest.quantiles = [0.2, 0.5, 0.8]
        forest.fit(boston_X, boston_y)
        pred = forest.predict(boston_X)
        assert len(pred) == boston_X.shape[0]

    def test_serialize(self, boston_X, boston_y):
        forest = GRFQuantileRegressor()
        forest.quantiles = [0.2, 0.5, 0.8]
        # not fitted
        tf = tempfile.TemporaryFile()
        pickle.dump(forest, tf)
        tf.seek(0)
        forest = pickle.load(tf)
        forest.fit(boston_X, boston_y)
        # fitted
        tf = tempfile.TemporaryFile()
        pickle.dump(forest, tf)
        tf.seek(0)
        new_forest = pickle.load(tf)
        pred = new_forest.predict(boston_X)
        assert len(pred) == boston_X.shape[0]

    def test_clone(self, boston_X, boston_y):
        forest = GRFQuantileRegressor()
        forest.quantiles = [0.2, 0.5, 0.8]
        forest.fit(boston_X, boston_y)
        clone(forest)

    def test_equalize_cluster_weights(
        self, boston_X, boston_y, boston_cluster, equalize_cluster_weights
    ):
        forest = GRFQuantileRegressor(equalize_cluster_weights=equalize_cluster_weights)
        forest.quantiles = [0.2, 0.5, 0.8]
        forest.fit(boston_X, boston_y, cluster=boston_cluster)
        if equalize_cluster_weights:
            assert forest.samples_per_cluster_ == 20
        else:
            assert forest.samples_per_cluster_ == boston_y.shape[0] - 20

        forest.fit(boston_X, boston_y, cluster=boston_cluster)
        forest.fit(boston_X, boston_y, cluster=None)
        assert forest.samples_per_cluster_ == 0

    def test_sample_fraction(self, boston_X, boston_y, sample_fraction):
        forest = GRFQuantileRegressor(sample_fraction=sample_fraction)
        forest.quantiles = [0.2, 0.5, 0.8]
        if sample_fraction <= 0 or sample_fraction > 1:
            with pytest.raises(ValueError):
                forest.fit(boston_X, boston_y)
        else:
            forest.fit(boston_X, boston_y)

    def test_mtry(self, boston_X, boston_y, mtry):
        forest = GRFQuantileRegressor(mtry=mtry)
        forest.quantiles = [0.2, 0.5, 0.8]
        forest.fit(boston_X, boston_y)
        if mtry is not None:
            assert forest.mtry_ == mtry
        else:
            assert forest.mtry_ == 6

    def test_honesty(self, boston_X, boston_y, honesty):
        forest = GRFQuantileRegressor(honesty=honesty)
        forest.quantiles = [0.2, 0.5, 0.8]
        forest.fit(boston_X, boston_y)

    def test_honesty_fraction(self, boston_X, boston_y, honesty_fraction):
        forest = GRFQuantileRegressor(
            honesty=True, honesty_fraction=honesty_fraction, honesty_prune_leaves=True
        )
        forest.quantiles = [0.2, 0.5, 0.8]
        if honesty_fraction <= 0 or honesty_fraction >= 1:
            with pytest.raises(RuntimeError):
                forest.fit(boston_X, boston_y)
        else:
            forest.fit(boston_X, boston_y)

    def test_honesty_prune_leaves(self, boston_X, boston_y, honesty_prune_leaves):
        forest = GRFQuantileRegressor(
            honesty=True, honesty_prune_leaves=honesty_prune_leaves
        )
        forest.quantiles = [0.2, 0.5, 0.8]
        forest.fit(boston_X, boston_y)

    def test_alpha(self, boston_X, boston_y, alpha):
        forest = GRFQuantileRegressor(alpha=alpha)
        forest.quantiles = [0.2, 0.5, 0.8]
        if alpha <= 0 or alpha >= 0.25:
            with pytest.raises(ValueError):
                forest.fit(boston_X, boston_y)
        else:
            forest.fit(boston_X, boston_y)

    def test_check_estimator(self):
        check_estimator(GRFQuantileRegressor(quantiles=[0.2]))

    def test_estimators_(self, boston_X, boston_y):
        forest = GRFQuantileRegressor(n_estimators=10, quantiles=[0.2])
        with pytest.raises(AttributeError):
            _ = forest.estimators_
        forest.fit(boston_X, boston_y)
        estimators = forest.estimators_
        assert len(estimators) == 10
        assert isinstance(estimators[0], GRFTreeQuantileRegressor)
        check_is_fitted(estimators[0])
