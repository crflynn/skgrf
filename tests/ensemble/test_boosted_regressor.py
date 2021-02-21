import pickle
import pytest
import tempfile
from sklearn import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from skgrf.ensemble.boosted_regressor import GRFBoostedRegressor


class TestGRFRegressor:
    def test_init(self):
        _ = GRFBoostedRegressor(tune_n_estimators=50)

    def test_fit(self, boston_X, boston_y):
        gbr = GRFBoostedRegressor(tune_n_estimators=50)
        with pytest.raises(NotFittedError):
            check_is_fitted(gbr)
        gbr.fit(boston_X, boston_y)
        check_is_fitted(gbr)
        assert hasattr(gbr, "boosted_forests_")
        assert hasattr(gbr, "mtry_")

    def test_predict(self, boston_X, boston_y):
        gbr = GRFBoostedRegressor(tune_n_estimators=50)
        gbr.fit(boston_X, boston_y)
        pred = gbr.predict(boston_X)
        assert len(pred) == boston_X.shape[0]

    def test_serialize(self, boston_X, boston_y):
        tf = tempfile.TemporaryFile()
        gbr = GRFBoostedRegressor(tune_n_estimators=50)
        gbr.fit(boston_X, boston_y)
        pickle.dump(gbr, tf)
        tf.seek(0)
        new_gbr = pickle.load(tf)
        pred = new_gbr.predict(boston_X)
        assert len(pred) == boston_X.shape[0]

    def test_clone(self, boston_X, boston_y):
        gbr = GRFBoostedRegressor(tune_n_estimators=50)
        gbr.fit(boston_X, boston_y)
        clone(gbr)
