import pickle
import tempfile

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted

from skgrf.ensemble import GRFForestQuantileRegressor
from skgrf.tree import GRFTreeQuantileRegressor


class TestGRFForestQuantileRegressor:
    def test_init(self):
        _ = GRFForestQuantileRegressor()

    def test_fit(self, boston_X, boston_y):
        forest = GRFForestQuantileRegressor()
        with pytest.raises(NotFittedError):
            check_is_fitted(forest)
        with pytest.raises(ValueError):
            forest.fit(boston_X, boston_y)
        forest.quantiles = [0.2, 0.5, 0.8]
        forest.fit(boston_X, boston_y)
        check_is_fitted(forest)
        assert hasattr(forest, "grf_forest_")
        assert hasattr(forest, "mtry_")
        assert forest.criterion == "gini"

    def test_predict(self, boston_X, boston_y):
        forest = GRFForestQuantileRegressor()
        forest.quantiles = [0.2, 0.5, 0.8]
        forest.fit(boston_X, boston_y)
        pred = forest.predict(boston_X)
        assert len(pred) == boston_X.shape[0]

    def test_with_X_nan(self, boston_X, boston_y):
        boston_X_nan = boston_X.copy()
        index = np.random.choice(boston_X_nan.size, 100, replace=False)
        boston_X_nan.ravel()[index] = np.nan
        assert np.sum(np.isnan(boston_X_nan)) == 100
        forest = GRFForestQuantileRegressor(quantiles=[0.5])
        forest.fit(boston_X_nan, boston_y)
        pred = forest.predict(boston_X_nan)
        assert len(pred) == boston_X_nan.shape[0]

    def test_serialize(self, boston_X, boston_y):
        forest = GRFForestQuantileRegressor()
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
        forest = GRFForestQuantileRegressor()
        forest.quantiles = [0.2, 0.5, 0.8]
        forest.fit(boston_X, boston_y)
        clone(forest)

    def test_equalize_cluster_weights(
        self, boston_X, boston_y, boston_cluster, equalize_cluster_weights
    ):
        forest = GRFForestQuantileRegressor(
            equalize_cluster_weights=equalize_cluster_weights
        )
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
        forest = GRFForestQuantileRegressor(sample_fraction=sample_fraction)
        forest.quantiles = [0.2, 0.5, 0.8]
        if sample_fraction <= 0 or sample_fraction > 1:
            with pytest.raises(ValueError):
                forest.fit(boston_X, boston_y)
        else:
            forest.fit(boston_X, boston_y)

    def test_mtry(self, boston_X, boston_y, mtry):
        forest = GRFForestQuantileRegressor(mtry=mtry)
        forest.quantiles = [0.2, 0.5, 0.8]
        forest.fit(boston_X, boston_y)
        if mtry is not None:
            assert forest.mtry_ == mtry
        else:
            assert forest.mtry_ == 6

    def test_honesty(self, boston_X, boston_y, honesty):
        forest = GRFForestQuantileRegressor(honesty=honesty)
        forest.quantiles = [0.2, 0.5, 0.8]
        forest.fit(boston_X, boston_y)

    def test_honesty_fraction(self, boston_X, boston_y, honesty_fraction):
        forest = GRFForestQuantileRegressor(
            honesty=True, honesty_fraction=honesty_fraction, honesty_prune_leaves=True
        )
        forest.quantiles = [0.2, 0.5, 0.8]
        if honesty_fraction <= 0 or honesty_fraction >= 1:
            with pytest.raises(RuntimeError):
                forest.fit(boston_X, boston_y)
        else:
            forest.fit(boston_X, boston_y)

    def test_honesty_prune_leaves(self, boston_X, boston_y, honesty_prune_leaves):
        forest = GRFForestQuantileRegressor(
            honesty=True, honesty_prune_leaves=honesty_prune_leaves
        )
        forest.quantiles = [0.2, 0.5, 0.8]
        forest.fit(boston_X, boston_y)

    def test_alpha(self, boston_X, boston_y, alpha):
        forest = GRFForestQuantileRegressor(alpha=alpha)
        forest.quantiles = [0.2, 0.5, 0.8]
        if alpha <= 0 or alpha >= 0.25:
            with pytest.raises(ValueError):
                forest.fit(boston_X, boston_y)
        else:
            forest.fit(boston_X, boston_y)

    def test_check_estimator(self):
        check_estimator(GRFForestQuantileRegressor(quantiles=[0.2]))

    def test_estimators_(self, boston_X, boston_y):
        forest = GRFForestQuantileRegressor(n_estimators=10, quantiles=[0.2])
        with pytest.raises(AttributeError):
            _ = forest.estimators_
        forest.fit(boston_X, boston_y)
        with pytest.raises(ValueError):
            _ = forest.estimators_
        forest = GRFForestQuantileRegressor(
            n_estimators=10, quantiles=[0.2], enable_tree_details=True
        )
        forest.fit(boston_X, boston_y)
        estimators = forest.estimators_
        assert len(estimators) == 10
        assert isinstance(estimators[0], GRFTreeQuantileRegressor)
        check_is_fitted(estimators[0])

    def test_get_estimator(self, boston_X, boston_y):
        forest = GRFForestQuantileRegressor(n_estimators=10, quantiles=[0.2])
        with pytest.raises(NotFittedError):
            _ = forest.get_estimator(idx=0)
        forest.fit(boston_X, boston_y)
        with pytest.raises(ValueError):
            _ = forest.get_estimator(idx=0)
        forest = GRFForestQuantileRegressor(
            n_estimators=10, quantiles=[0.2], enable_tree_details=True
        )
        forest.fit(boston_X, boston_y)
        estimator = forest.get_estimator(0)
        check_is_fitted(estimator)
        assert isinstance(estimator, GRFTreeQuantileRegressor)
        with pytest.raises(IndexError):
            _ = forest.get_estimator(idx=20)

    def test_get_split_frequencies(self, boston_X, boston_y):
        forest = GRFForestQuantileRegressor(quantiles=[0.2])
        forest.fit(boston_X, boston_y)
        sf = forest.get_split_frequencies()
        assert sf.shape[1] == boston_X.shape[1]

    def test_get_feature_importances(self, boston_X, boston_y):
        forest = GRFForestQuantileRegressor(quantiles=[0.2])
        forest.fit(boston_X, boston_y)
        fi = forest.get_feature_importances()
        assert len(fi) == boston_X.shape[1]

    def test_get_kernel_weights(self, boston_X, boston_y):
        X_train, X_test, y_train, y_test = train_test_split(
            boston_X, boston_y, test_size=0.33, random_state=42
        )
        forest = GRFForestQuantileRegressor(quantiles=[0.2])
        forest.fit(X_train, y_train)
        weights = forest.get_kernel_weights(X_test)
        assert weights.shape[0] == X_test.shape[0]
        assert weights.shape[1] == X_train.shape[0]
        oob_weights = forest.get_kernel_weights(X_train, True)
        assert oob_weights.shape[0] == X_train.shape[0]
        assert oob_weights.shape[1] == X_train.shape[0]
