import numpy as np
import pickle
import pytest
import tempfile
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted

from skgrf.ensemble import GRFRegressor
from skgrf.tree.regressor import GRFTreeRegressor


class TestGRFTreeRegressor:
    def test_init(self):
        _ = GRFTreeRegressor()

    def test_fit(self, boston_X, boston_y):
        gfr = GRFTreeRegressor()
        with pytest.raises(NotFittedError):
            check_is_fitted(gfr)
        gfr.fit(boston_X, boston_y)
        check_is_fitted(gfr)
        assert hasattr(gfr, "grf_forest_")
        assert hasattr(gfr, "mtry_")
        assert gfr.grf_forest_["_num_trees"] == 1

    def test_predict(self, boston_X, boston_y):
        gfr = GRFTreeRegressor()
        gfr.fit(boston_X, boston_y)
        pred = gfr.predict(boston_X)
        assert len(pred) == boston_X.shape[0]

    def test_predict_oob(self, boston_X, boston_y):
        gfr = GRFTreeRegressor()
        gfr.fit(boston_X, boston_y, compute_oob_predictions=True)
        pred = np.atleast_1d(np.squeeze(np.array(gfr.grf_forest_["predictions"])))
        assert len(pred) == boston_X.shape[0]

    def test_serialize(self, boston_X, boston_y):
        gfr = GRFTreeRegressor()
        # not fitted
        tf = tempfile.TemporaryFile()
        pickle.dump(gfr, tf)
        tf.seek(0)
        gfr = pickle.load(tf)
        gfr.fit(boston_X, boston_y)
        # fitted
        tf = tempfile.TemporaryFile()
        pickle.dump(gfr, tf)
        tf.seek(0)
        new_gfr = pickle.load(tf)
        pred = new_gfr.predict(boston_X)
        assert len(pred) == boston_X.shape[0]

    def test_clone(self, boston_X, boston_y):
        gfr = GRFTreeRegressor()
        gfr.fit(boston_X, boston_y)
        clone(gfr)

    def test_equalize_cluster_weights(
        self, boston_X, boston_y, boston_cluster, equalize_cluster_weights
    ):
        gfr = GRFTreeRegressor(equalize_cluster_weights=equalize_cluster_weights)
        gfr.fit(boston_X, boston_y, cluster=boston_cluster)
        if equalize_cluster_weights:
            assert gfr.samples_per_cluster_ == 20
        else:
            assert gfr.samples_per_cluster_ == boston_y.shape[0] - 20

        if equalize_cluster_weights:
            with pytest.raises(ValueError):
                gfr.fit(
                    boston_X, boston_y, cluster=boston_cluster, sample_weight=boston_y
                )

        gfr.fit(boston_X, boston_y, cluster=None)
        assert gfr.samples_per_cluster_ == 0

    def test_sample_fraction(
        self, boston_X, boston_y, sample_fraction
    ):  # and ci_group_size
        gfr = GRFTreeRegressor(sample_fraction=sample_fraction)
        if sample_fraction <= 0 or sample_fraction > 1:
            with pytest.raises(ValueError):
                gfr.fit(boston_X, boston_y)
        else:
            gfr.fit(boston_X, boston_y)

    def test_mtry(self, boston_X, boston_y, mtry):
        gfr = GRFTreeRegressor(mtry=mtry)
        gfr.fit(boston_X, boston_y)
        if mtry is not None:
            assert gfr.mtry_ == mtry
        else:
            assert gfr.mtry_ == 6

    def test_honesty(self, boston_X, boston_y, honesty):
        gfr = GRFTreeRegressor(honesty=honesty)
        gfr.fit(boston_X, boston_y)

    def test_honesty_fraction(self, boston_X, boston_y, honesty_fraction):
        gfr = GRFTreeRegressor(
            honesty=True, honesty_fraction=honesty_fraction, honesty_prune_leaves=True
        )
        if honesty_fraction <= 0 or honesty_fraction >= 1:
            with pytest.raises(RuntimeError):
                gfr.fit(boston_X, boston_y)
        else:
            gfr.fit(boston_X, boston_y)

    def test_honesty_prune_leaves(self, boston_X, boston_y, honesty_prune_leaves):
        gfr = GRFTreeRegressor(honesty=True, honesty_prune_leaves=honesty_prune_leaves)
        gfr.fit(boston_X, boston_y)

    def test_alpha(self, boston_X, boston_y, alpha):
        gfr = GRFTreeRegressor(alpha=alpha)
        if alpha <= 0 or alpha >= 0.25:
            with pytest.raises(ValueError):
                gfr.fit(boston_X, boston_y)
        else:
            gfr.fit(boston_X, boston_y)

    def test_check_estimator(self):
        check_estimator(GRFTreeRegressor())

    def test_from_forest(self, boston_X, boston_y):
        gfr = GRFRegressor()
        gfr.fit(boston_X, boston_y)
        tree = GRFTreeRegressor.from_forest(forest=gfr, idx=0)
        tree.predict(boston_X)
