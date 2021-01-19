import numpy as np
import pickle
import pytest
import random
import tempfile
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from skgrf.ensemble import GRFQuantileRegressor


class TestGRFQuantileRegressor:
    def test_init(self):
        _ = GRFQuantileRegressor()

    def test_fit(self, boston_X, boston_y):
        gqr = GRFQuantileRegressor()
        with pytest.raises(NotFittedError):
            check_is_fitted(gqr)
        with pytest.raises(ValueError):
            gqr.fit(boston_X, boston_y)
        gqr.quantiles = [0.2, 0.5, 0.8]
        gqr.fit(boston_X, boston_y)
        check_is_fitted(gqr)
        assert hasattr(gqr, "grf_forest_")
        assert hasattr(gqr, "mtry_")

    def test_predict(self, boston_X, boston_y):
        gqr = GRFQuantileRegressor()
        gqr.quantiles = [0.2, 0.5, 0.8]
        gqr.fit(boston_X, boston_y)
        pred = gqr.predict(boston_X)
        assert len(pred) == boston_X.shape[0]

    def test_serialize(self, boston_X, boston_y):
        tf = tempfile.TemporaryFile()
        gqr = GRFQuantileRegressor()
        gqr.quantiles = [0.2, 0.5, 0.8]
        gqr.fit(boston_X, boston_y)
        pickle.dump(gqr, tf)
        tf.seek(0)
        new_gqr = pickle.load(tf)
        pred = new_gqr.predict(boston_X)
        assert len(pred) == boston_X.shape[0]

    def test_clone(self, boston_X, boston_y):
        gqr = GRFQuantileRegressor()
        gqr.quantiles = [0.2, 0.5, 0.8]
        gqr.fit(boston_X, boston_y)
        clone(gqr)
