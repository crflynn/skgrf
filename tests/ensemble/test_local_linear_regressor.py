import pickle
import pytest
import tempfile
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator
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
        glr = GRFLocalLinearRegressor(ll_split_cutoff=0)
        # not fitted
        tf = tempfile.TemporaryFile()
        pickle.dump(glr, tf)
        tf.seek(0)
        glr = pickle.load(tf)
        glr.fit(boston_X, boston_y)
        # fitted
        tf = tempfile.TemporaryFile()
        pickle.dump(glr, tf)
        tf.seek(0)
        new_glr = pickle.load(tf)
        pred = new_glr.predict(boston_X)
        assert len(pred) == boston_X.shape[0]

    def test_clone(self, boston_X, boston_y):
        glr = GRFLocalLinearRegressor(ll_split_cutoff=0)
        glr.fit(boston_X, boston_y)
        clone(glr)

    def test_equalize_cluster_weights(
        self, boston_X, boston_y, boston_cluster, equalize_cluster_weights
    ):
        glr = GRFLocalLinearRegressor(equalize_cluster_weights=equalize_cluster_weights)
        glr.fit(boston_X, boston_y, cluster=boston_cluster)
        if equalize_cluster_weights:
            assert glr.samples_per_cluster_ == 20
        else:
            assert glr.samples_per_cluster_ == boston_y.shape[0] - 20

        if equalize_cluster_weights:
            with pytest.raises(ValueError):
                glr.fit(
                    boston_X, boston_y, cluster=boston_cluster, sample_weight=boston_y
                )

        glr.fit(boston_X, boston_y, cluster=None)
        assert glr.samples_per_cluster_ == 0

    def test_sample_fraction(
        self, boston_X, boston_y, sample_fraction
    ):  # and ci_group_size
        glr = GRFLocalLinearRegressor(sample_fraction=sample_fraction, ci_group_size=1)
        if sample_fraction <= 0 or sample_fraction > 1:
            with pytest.raises(ValueError):
                glr.fit(boston_X, boston_y)
        else:
            glr.fit(boston_X, boston_y)

        glr = GRFLocalLinearRegressor(sample_fraction=sample_fraction, ci_group_size=2)
        if sample_fraction <= 0 or sample_fraction > 0.5:
            with pytest.raises(ValueError):
                glr.fit(boston_X, boston_y)
        else:
            glr.fit(boston_X, boston_y)

    def test_mtry(self, boston_X, boston_y, mtry):
        glr = GRFLocalLinearRegressor(mtry=mtry)
        glr.fit(boston_X, boston_y)
        if mtry is not None:
            assert glr.mtry_ == mtry
        else:
            assert glr.mtry_ == 6

    def test_honesty(self, boston_X, boston_y, honesty):
        glr = GRFLocalLinearRegressor(honesty=honesty)
        glr.fit(boston_X, boston_y)

    def test_honesty_fraction(self, boston_X, boston_y, honesty_fraction):
        glr = GRFLocalLinearRegressor(
            honesty=True, honesty_fraction=honesty_fraction, honesty_prune_leaves=True
        )
        if honesty_fraction <= 0 or honesty_fraction >= 1:
            with pytest.raises(RuntimeError):
                glr.fit(boston_X, boston_y)
        else:
            glr.fit(boston_X, boston_y)

    def test_honesty_prune_leaves(self, boston_X, boston_y, honesty_prune_leaves):
        glr = GRFLocalLinearRegressor(
            honesty=True, honesty_prune_leaves=honesty_prune_leaves
        )
        glr.fit(boston_X, boston_y)

    def test_alpha(self, boston_X, boston_y, alpha):
        glr = GRFLocalLinearRegressor(alpha=alpha)
        if alpha <= 0 or alpha >= 0.25:
            with pytest.raises(ValueError):
                glr.fit(boston_X, boston_y)
        else:
            glr.fit(boston_X, boston_y)

    def test_check_estimator(self):
        check_estimator(GRFLocalLinearRegressor())
