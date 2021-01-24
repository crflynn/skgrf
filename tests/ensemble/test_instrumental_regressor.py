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
