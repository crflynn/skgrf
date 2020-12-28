import numpy as np
import pickle
import pytest
import random
import tempfile
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
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
