import pickle
import pytest
import tempfile
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from skgrf.ensemble import GRFLocalLinearRegressor


class TestGRFLocalLinearRegressor:
    def test_init(self):
        _ = GRFLocalLinearRegressor()

    def test_fit(self, boston_X, boston_y):
        glr = GRFLocalLinearRegressor(ll_split_cutoff=0)
        with pytest.raises(NotFittedError):
            check_is_fitted(glr)
        glr.fit(boston_X, boston_y)
        check_is_fitted(glr)
        assert hasattr(glr, "grf_forest_")
        assert hasattr(glr, "mtry_")

    def test_predict(self, boston_X, boston_y):
        glr = GRFLocalLinearRegressor(ll_split_cutoff=0)
        glr.fit(boston_X, boston_y)
        pred = glr.predict(boston_X)
        assert len(pred) == boston_X.shape[0]

    def test_serialize(self, boston_X, boston_y):
        tf = tempfile.TemporaryFile()
        glr = GRFLocalLinearRegressor(ll_split_cutoff=0)
        glr.fit(boston_X, boston_y)
        pickle.dump(glr, tf)
        tf.seek(0)
        new_glr = pickle.load(tf)
        pred = new_glr.predict(boston_X)
        assert len(pred) == boston_X.shape[0]

    def test_clone(self, boston_X, boston_y):
        glr = GRFLocalLinearRegressor(ll_split_cutoff=0)
        glr.fit(boston_X, boston_y)
        clone(glr)
