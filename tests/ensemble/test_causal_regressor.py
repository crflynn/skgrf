import pickle
import pytest
import tempfile
from sklearn import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from skgrf.ensemble import GRFCausalRegressor


class TestGRFCausal:
    def test_init(self):
        _ = GRFCausalRegressor()

    def test_fit(self, boston_X, boston_y):
        gfc = GRFCausalRegressor()
        with pytest.raises(NotFittedError):
            check_is_fitted(gfc)
        gfc.fit(boston_X, boston_y, boston_y)
        check_is_fitted(gfc)
        assert hasattr(gfc, "grf_forest_")
        assert hasattr(gfc, "mtry_")

    def test_predict(self, boston_X, boston_y):
        gfc = GRFCausalRegressor()
        gfc.fit(boston_X, boston_y, boston_y)
        pred = gfc.predict(boston_X)
        assert len(pred) == boston_X.shape[0]

    def test_serialize(self, boston_X, boston_y):
        tf = tempfile.TemporaryFile()
        gfc = GRFCausalRegressor()
        gfc.fit(boston_X, boston_y, boston_y)
        pickle.dump(gfc, tf)
        tf.seek(0)
        new_gfc = pickle.load(tf)
        pred = new_gfc.predict(boston_X)
        assert len(pred) == boston_X.shape[0]

    def test_clone(self, boston_X, boston_y):
        gfc = GRFCausalRegressor()
        gfc.fit(boston_X, boston_y, boston_y)
        clone(gfc)
