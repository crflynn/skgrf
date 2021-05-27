import pickle
import pytest
import tempfile
from sklearn import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from skgrf.ensemble.causal_regressor import GRFCausalRegressor
from skgrf.tree.causal_regressor import GRFTreeCausalRegressor


class TestGRFCausalRegressor:
    def test_init(self):
        _ = GRFCausalRegressor(n_estimators=100)

    def test_fit(self, causal_X, causal_y, causal_w):
        gfc = GRFCausalRegressor(n_estimators=100)
        with pytest.raises(NotFittedError):
            check_is_fitted(gfc)
        gfc.fit(causal_X, causal_y, causal_w)
        check_is_fitted(gfc)
        assert hasattr(gfc, "grf_forest_")
        assert hasattr(gfc, "mtry_")

    def test_predict(self, causal_X, causal_y, causal_w):
        gfc = GRFCausalRegressor(n_estimators=100)
        gfc.fit(causal_X, causal_y, causal_w)
        pred = gfc.predict(causal_X)
        assert len(pred) == causal_X.shape[0]

    def test_serialize(self, causal_X, causal_y, causal_w):
        gfc = GRFCausalRegressor(n_estimators=100)
        # not fitted
        tf = tempfile.TemporaryFile()
        pickle.dump(gfc, tf)
        tf.seek(0)
        gfc = pickle.load(tf)
        gfc.fit(causal_X, causal_y, causal_w)
        # fitted
        tf = tempfile.TemporaryFile()
        pickle.dump(gfc, tf)
        tf.seek(0)
        new_gfc = pickle.load(tf)
        pred = new_gfc.predict(causal_X)
        assert len(pred) == causal_X.shape[0]

    def test_clone(self, causal_X, causal_y, causal_w):
        gfc = GRFCausalRegressor(n_estimators=100)
        gfc.fit(causal_X, causal_y, causal_w)
        clone(gfc)

    def test_equalize_cluster_weights(
        self,
        causal_X,
        causal_y,
        causal_w,
        causal_cluster,
        equalize_cluster_weights,
    ):
        gcr = GRFCausalRegressor(
            n_estimators=100, equalize_cluster_weights=equalize_cluster_weights
        )
        gcr.fit(causal_X, causal_y, causal_w, causal_w, cluster=causal_cluster)
        if equalize_cluster_weights:
            assert gcr.samples_per_cluster_ == 20
        else:
            assert gcr.samples_per_cluster_ == causal_y.shape[0] - 20

        if equalize_cluster_weights:
            with pytest.raises(ValueError):
                gcr.fit(
                    causal_X,
                    causal_y,
                    causal_w,
                    causal_w,
                    cluster=causal_cluster,
                    sample_weight=causal_y,
                )

        gcr.fit(causal_X, causal_y, causal_w, causal_w, cluster=None)
        assert gcr.samples_per_cluster_ == 0

    def test_sample_fraction(
        self, causal_X, causal_y, causal_w, sample_fraction
    ):  # and ci_group_size
        gcr = GRFCausalRegressor(
            n_estimators=100, sample_fraction=sample_fraction, ci_group_size=1
        )
        if sample_fraction <= 0 or sample_fraction > 1:
            with pytest.raises(ValueError):
                gcr.fit(causal_X, causal_y, causal_w, causal_w)
        else:
            gcr.fit(causal_X, causal_y, causal_w, causal_w)

        gcr = GRFCausalRegressor(
            n_estimators=100, sample_fraction=sample_fraction, ci_group_size=2
        )
        if sample_fraction <= 0 or sample_fraction > 0.5:
            with pytest.raises(ValueError):
                gcr.fit(causal_X, causal_y, causal_w, causal_w)
        else:
            gcr.fit(causal_X, causal_y, causal_w, causal_w)

    def test_mtry(self, causal_X, causal_y, causal_w, mtry):
        gcr = GRFCausalRegressor(n_estimators=100, mtry=mtry)
        gcr.fit(causal_X, causal_y, causal_w, causal_w)
        if mtry is not None:
            assert gcr.mtry_ == mtry
        else:
            assert gcr.mtry_ == 5

    def test_honesty(self, causal_X, causal_y, causal_w, honesty):
        gcr = GRFCausalRegressor(n_estimators=100, honesty=honesty)
        gcr.fit(causal_X, causal_y, causal_w, causal_w)

    def test_honesty_fraction(self, causal_X, causal_y, causal_w, honesty_fraction):
        gcr = GRFCausalRegressor(
            n_estimators=100,
            honesty=True,
            honesty_fraction=honesty_fraction,
            honesty_prune_leaves=True,
        )
        if honesty_fraction <= 0 or honesty_fraction >= 1:
            with pytest.raises(RuntimeError):
                gcr.fit(causal_X, causal_y, causal_w, causal_w)
        else:
            gcr.fit(causal_X, causal_y, causal_w, causal_w)

    def test_honesty_prune_leaves(
        self, causal_X, causal_y, causal_w, honesty_prune_leaves
    ):
        gcr = GRFCausalRegressor(
            n_estimators=100, honesty=True, honesty_prune_leaves=honesty_prune_leaves
        )
        gcr.fit(causal_X, causal_y, causal_w, causal_w)

    def test_alpha(self, causal_X, causal_y, causal_w, alpha):
        gcr = GRFCausalRegressor(n_estimators=100, alpha=alpha)
        if alpha <= 0 or alpha >= 0.25:
            with pytest.raises(ValueError):
                gcr.fit(causal_X, causal_y, causal_w, causal_w)
        else:
            gcr.fit(causal_X, causal_y, causal_w, causal_w)

    def test_orthogonal_boosting(
        self, causal_X, causal_y, causal_w, orthogonal_boosting
    ):
        gcr = GRFCausalRegressor(
            n_estimators=100, orthogonal_boosting=orthogonal_boosting
        )
        gcr.fit(causal_X, causal_y, causal_w)

    # cant use this because of extra required fit params
    # def test_check_estimator(self):
    #     check_estimator(GRFCausalRegressor())

    def test_estimators_(self, causal_X, causal_y, causal_w):
        gcr = GRFCausalRegressor(n_estimators=10)
        with pytest.raises(AttributeError):
            _ = gcr.estimators_
        gcr.fit(causal_X, causal_y, causal_w, causal_w)
        estimators = gcr.estimators_
        assert len(estimators) == 10
        assert isinstance(estimators[0], GRFTreeCausalRegressor)
        check_is_fitted(estimators[0])
