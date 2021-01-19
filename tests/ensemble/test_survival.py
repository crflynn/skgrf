import pickle
import pytest
import tempfile
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from skgrf.ensemble import GRFSurvival


class TestGRFSurvival:
    def test_init(self):
        _ = GRFSurvival()

    def test_fit(self, lung_X, lung_y):
        gfs = GRFSurvival()
        with pytest.raises(NotFittedError):
            check_is_fitted(gfs)
        gfs.fit(lung_X, lung_y)
        check_is_fitted(gfs)
        assert hasattr(gfs, "grf_forest_")
        assert hasattr(gfs, "mtry_")

    def test_predict(self, lung_X, lung_y):
        gfs = GRFSurvival()
        gfs.fit(lung_X, lung_y)
        pred = gfs.predict(lung_X)
        assert len(pred) == lung_X.shape[0]

    def test_serialize(self, lung_X, lung_y):
        tf = tempfile.TemporaryFile()
        gfs = GRFSurvival()
        gfs.fit(lung_X, lung_y)
        pickle.dump(gfs, tf)
        tf.seek(0)
        new_gfs = pickle.load(tf)
        pred = new_gfs.predict(lung_X)
        assert len(pred) == lung_X.shape[0]

    def test_clone(self, lung_X, lung_y):
        gfs = GRFSurvival()
        gfs.fit(lung_X, lung_y)
        clone(gfs)
