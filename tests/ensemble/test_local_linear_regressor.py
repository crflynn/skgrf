import pickle
import pytest
import tempfile
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted

from skgrf.ensemble import GRFLocalLinearRegressor
from skgrf.tree import GRFTreeLocalLinearRegressor


class TestGRFLocalLinearRegressor:
    def test_init(self):
        _ = GRFLocalLinearRegressor()

    def test_fit(self, boston_X, boston_y):
        forest = GRFLocalLinearRegressor(ll_split_cutoff=0)
        with pytest.raises(NotFittedError):
            check_is_fitted(forest)
        forest.fit(boston_X, boston_y)
        check_is_fitted(forest)
        assert hasattr(forest, "grf_forest_")
        assert hasattr(forest, "mtry_")

    def test_predict(self, boston_X, boston_y):
        forest = GRFLocalLinearRegressor(ll_split_cutoff=0)
        forest.fit(boston_X, boston_y)
        pred = forest.predict(boston_X)
        assert len(pred) == boston_X.shape[0]

    def test_serialize(self, boston_X, boston_y):
        forest = GRFLocalLinearRegressor(ll_split_cutoff=0)
        # not fitted
        tf = tempfile.TemporaryFile()
        pickle.dump(forest, tf)
        tf.seek(0)
        forest = pickle.load(tf)
        forest.fit(boston_X, boston_y)
        # fitted
        tf = tempfile.TemporaryFile()
        pickle.dump(forest, tf)
        tf.seek(0)
        new_forest = pickle.load(tf)
        pred = new_forest.predict(boston_X)
        assert len(pred) == boston_X.shape[0]

    def test_clone(self, boston_X, boston_y):
        forest = GRFLocalLinearRegressor(ll_split_cutoff=0)
        forest.fit(boston_X, boston_y)
        clone(forest)

    def test_equalize_cluster_weights(
        self, boston_X, boston_y, boston_cluster, equalize_cluster_weights
    ):
        forest = GRFLocalLinearRegressor(
            equalize_cluster_weights=equalize_cluster_weights
        )
        forest.fit(boston_X, boston_y, cluster=boston_cluster)
        if equalize_cluster_weights:
            assert forest.samples_per_cluster_ == 20
        else:
            assert forest.samples_per_cluster_ == boston_y.shape[0] - 20

        if equalize_cluster_weights:
            with pytest.raises(ValueError):
                forest.fit(
                    boston_X, boston_y, cluster=boston_cluster, sample_weight=boston_y
                )

        forest.fit(boston_X, boston_y, cluster=None)
        assert forest.samples_per_cluster_ == 0

    def test_sample_fraction(
        self, boston_X, boston_y, sample_fraction
    ):  # and ci_group_size
        forest = GRFLocalLinearRegressor(
            sample_fraction=sample_fraction, ci_group_size=1
        )
        if sample_fraction <= 0 or sample_fraction > 1:
            with pytest.raises(ValueError):
                forest.fit(boston_X, boston_y)
        else:
            forest.fit(boston_X, boston_y)

        forest = GRFLocalLinearRegressor(
            sample_fraction=sample_fraction, ci_group_size=2
        )
        if sample_fraction <= 0 or sample_fraction > 0.5:
            with pytest.raises(ValueError):
                forest.fit(boston_X, boston_y)
        else:
            forest.fit(boston_X, boston_y)

    def test_mtry(self, boston_X, boston_y, mtry):
        forest = GRFLocalLinearRegressor(mtry=mtry)
        forest.fit(boston_X, boston_y)
        if mtry is not None:
            assert forest.mtry_ == mtry
        else:
            assert forest.mtry_ == 6

    def test_honesty(self, boston_X, boston_y, honesty):
        forest = GRFLocalLinearRegressor(honesty=honesty)
        forest.fit(boston_X, boston_y)

    def test_honesty_fraction(self, boston_X, boston_y, honesty_fraction):
        forest = GRFLocalLinearRegressor(
            honesty=True, honesty_fraction=honesty_fraction, honesty_prune_leaves=True
        )
        if honesty_fraction <= 0 or honesty_fraction >= 1:
            with pytest.raises(RuntimeError):
                forest.fit(boston_X, boston_y)
        else:
            forest.fit(boston_X, boston_y)

    def test_honesty_prune_leaves(self, boston_X, boston_y, honesty_prune_leaves):
        forest = GRFLocalLinearRegressor(
            honesty=True, honesty_prune_leaves=honesty_prune_leaves
        )
        forest.fit(boston_X, boston_y)

    def test_alpha(self, boston_X, boston_y, alpha):
        forest = GRFLocalLinearRegressor(alpha=alpha)
        if alpha <= 0 or alpha >= 0.25:
            with pytest.raises(ValueError):
                forest.fit(boston_X, boston_y)
        else:
            forest.fit(boston_X, boston_y)

    def test_check_estimator(self):
        check_estimator(GRFLocalLinearRegressor())

    def test_estimators_(self, boston_X, boston_y):
        forest = GRFLocalLinearRegressor(n_estimators=10)
        with pytest.raises(AttributeError):
            _ = forest.estimators_
        forest.fit(boston_X, boston_y)
        estimators = forest.estimators_
        assert len(estimators) == 10
        assert isinstance(estimators[0], GRFTreeLocalLinearRegressor)
        check_is_fitted(estimators[0])

    def test_get_split_frequencies(self, boston_X, boston_y):
        forest = GRFLocalLinearRegressor()
        forest.fit(boston_X, boston_y)
        sf = forest.get_split_frequencies()
        assert sf.shape[1] == boston_X.shape[1]

    def test_get_feature_importances(self, boston_X, boston_y):
        forest = GRFLocalLinearRegressor()
        forest.fit(boston_X, boston_y)
        fi = forest.get_feature_importances()
        assert len(fi) == boston_X.shape[1]

    def test_get_kernel_weights(self, boston_X, boston_y):
        X_train, X_test, y_train, y_test = train_test_split(
            boston_X, boston_y, test_size=0.33, random_state=42
        )
        forest = GRFLocalLinearRegressor()
        forest.fit(X_train, y_train)
        weights = forest.get_kernel_weights(X_test)
        assert weights.shape[0] == X_test.shape[0]
        assert weights.shape[1] == X_train.shape[0]
        oob_weights = forest.get_kernel_weights(X_train, True)
        assert oob_weights.shape[0] == X_train.shape[0]
        assert oob_weights.shape[1] == X_train.shape[0]
