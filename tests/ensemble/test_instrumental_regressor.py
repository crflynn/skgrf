import pickle
import pytest
import tempfile
from sklearn import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from skgrf.ensemble import GRFInstrumentalRegressor
from skgrf.tree import GRFTreeInstrumentalRegressor


class TestGRFInstrumentalRegressor:
    def test_init(self):
        _ = GRFInstrumentalRegressor()

    def test_fit(self, causal_X, causal_y, causal_w):
        forest = GRFInstrumentalRegressor()
        with pytest.raises(NotFittedError):
            check_is_fitted(forest)
        forest.fit(causal_X, causal_y, causal_w, causal_w)
        check_is_fitted(forest)
        assert hasattr(forest, "grf_forest_")
        assert hasattr(forest, "mtry_")

    def test_predict(self, causal_X, causal_y, causal_w):
        forest = GRFInstrumentalRegressor()
        forest.fit(causal_X, causal_y, causal_w, causal_w)
        pred = forest.predict(causal_X)
        assert len(pred) == causal_X.shape[0]

    def test_serialize(self, causal_X, causal_y, causal_w):
        forest = GRFInstrumentalRegressor()
        # not fitted
        tf = tempfile.TemporaryFile()
        pickle.dump(forest, tf)
        tf.seek(0)
        forest = pickle.load(tf)
        forest.fit(causal_X, causal_y, causal_w, causal_w)
        # fitted
        tf = tempfile.TemporaryFile()
        pickle.dump(forest, tf)
        tf.seek(0)
        new_forest = pickle.load(tf)
        pred = new_forest.predict(causal_X)
        assert len(pred) == causal_X.shape[0]

    def test_clone(self, causal_X, causal_y, causal_w):
        forest = GRFInstrumentalRegressor()
        forest.fit(causal_X, causal_y, causal_w, causal_w)
        clone(forest)

    def test_equalize_cluster_weights(
        self,
        causal_X,
        causal_y,
        causal_w,
        causal_cluster,
        equalize_cluster_weights,
    ):
        forest = GRFInstrumentalRegressor(
            equalize_cluster_weights=equalize_cluster_weights
        )
        forest.fit(causal_X, causal_y, causal_w, causal_w, cluster=causal_cluster)
        if equalize_cluster_weights:
            assert forest.samples_per_cluster_ == 20
        else:
            assert forest.samples_per_cluster_ == causal_y.shape[0] - 20

        if equalize_cluster_weights:
            with pytest.raises(ValueError):
                forest.fit(
                    causal_X,
                    causal_y,
                    causal_w,
                    causal_w,
                    cluster=causal_cluster,
                    sample_weight=causal_y,
                )

        forest.fit(causal_X, causal_y, causal_w, causal_w, cluster=None)
        assert forest.samples_per_cluster_ == 0

    def test_sample_fraction(
        self, causal_X, causal_y, causal_w, sample_fraction
    ):  # and ci_group_size
        forest = GRFInstrumentalRegressor(
            sample_fraction=sample_fraction, ci_group_size=1
        )
        if sample_fraction <= 0 or sample_fraction > 1:
            with pytest.raises(ValueError):
                forest.fit(causal_X, causal_y, causal_w, causal_w)
        else:
            forest.fit(causal_X, causal_y, causal_w, causal_w)

        forest = GRFInstrumentalRegressor(
            sample_fraction=sample_fraction, ci_group_size=2
        )
        if sample_fraction <= 0 or sample_fraction > 0.5:
            with pytest.raises(ValueError):
                forest.fit(causal_X, causal_y, causal_w, causal_w)
        else:
            forest.fit(causal_X, causal_y, causal_w, causal_w)

    def test_mtry(self, causal_X, causal_y, causal_w, mtry):
        forest = GRFInstrumentalRegressor(mtry=mtry)
        forest.fit(causal_X, causal_y, causal_w, causal_w)
        if mtry is not None:
            assert forest.mtry_ == mtry
        else:
            assert forest.mtry_ == 5

    def test_honesty(self, causal_X, causal_y, causal_w, honesty):
        forest = GRFInstrumentalRegressor(honesty=honesty)
        forest.fit(causal_X, causal_y, causal_w, causal_w)

    def test_honesty_fraction(self, causal_X, causal_y, causal_w, honesty_fraction):
        forest = GRFInstrumentalRegressor(
            honesty=True, honesty_fraction=honesty_fraction, honesty_prune_leaves=True
        )
        if honesty_fraction <= 0 or honesty_fraction >= 1:
            with pytest.raises(RuntimeError):
                forest.fit(causal_X, causal_y, causal_w, causal_w)
        else:
            forest.fit(causal_X, causal_y, causal_w, causal_w)

    def test_honesty_prune_leaves(
        self, causal_X, causal_y, causal_w, honesty_prune_leaves
    ):
        forest = GRFInstrumentalRegressor(
            honesty=True, honesty_prune_leaves=honesty_prune_leaves
        )
        forest.fit(causal_X, causal_y, causal_w, causal_w)

    def test_alpha(self, causal_X, causal_y, causal_w, alpha):
        forest = GRFInstrumentalRegressor(alpha=alpha)
        if alpha <= 0 or alpha >= 0.25:
            with pytest.raises(ValueError):
                forest.fit(causal_X, causal_y, causal_w, causal_w)
        else:
            forest.fit(causal_X, causal_y, causal_w, causal_w)

    # cant use this because of extra required fit params
    # def test_check_estimator(self):
    #     check_estimator(GRFInstrumentalRegressor())

    def test_estimators_(self, causal_X, causal_y, causal_w):
        forest = GRFInstrumentalRegressor(n_estimators=10)
        with pytest.raises(AttributeError):
            _ = forest.estimators_
        forest.fit(causal_X, causal_y, causal_w, causal_w)
        estimators = forest.estimators_
        assert len(estimators) == 10
        assert isinstance(estimators[0], GRFTreeInstrumentalRegressor)
        check_is_fitted(estimators[0])

    def test_get_split_frequencies(self, causal_X, causal_y, causal_w):
        forest = GRFInstrumentalRegressor()
        forest.fit(causal_X, causal_y, causal_w, causal_w)
        sf = forest.get_split_frequencies()
        assert sf.shape[1] == causal_X.shape[1]

    def test_get_feature_importances(self, causal_X, causal_y, causal_w):
        forest = GRFInstrumentalRegressor()
        forest.fit(causal_X, causal_y, causal_w, causal_w)
        fi = forest.get_feature_importances()
        assert len(fi) == causal_X.shape[1]
