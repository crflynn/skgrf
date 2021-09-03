import numpy as np
import pickle
import pytest
import tempfile
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from skgrf.ensemble.survival import GRFForestSurvival
from skgrf.tree.survival import GRFTreeSurvival


class TestGRFForestSurvival:
    def test_init(self):
        _ = GRFForestSurvival()

    def test_fit(self, lung_X, lung_y):
        forest = GRFForestSurvival()
        with pytest.raises(NotFittedError):
            check_is_fitted(forest)
        forest.fit(lung_X, lung_y)
        check_is_fitted(forest)
        assert hasattr(forest, "grf_forest_")
        assert hasattr(forest, "mtry_")
        assert forest.criterion == "logrank"

    def test_predict(self, lung_X, lung_y):
        forest = GRFForestSurvival()
        forest.fit(lung_X, lung_y)
        pred = forest.predict(lung_X)
        assert len(pred) == lung_X.shape[0]

    def test_with_X_nan(self, lung_X, lung_y):
        index = np.random.choice(lung_X.size, 100, replace=False)
        lung_X.ravel()[index] = np.nan
        forest = GRFForestSurvival()
        forest.fit(lung_X, lung_y)
        pred = forest.predict(lung_X)
        assert len(pred) == lung_X.shape[0]

    def test_serialize(self, lung_X, lung_y):
        forest = GRFForestSurvival()
        # not fitted
        tf = tempfile.TemporaryFile()
        pickle.dump(forest, tf)
        tf.seek(0)
        forest = pickle.load(tf)
        forest.fit(lung_X, lung_y)
        # fitted
        tf = tempfile.TemporaryFile()
        pickle.dump(forest, tf)
        tf.seek(0)
        new_forest = pickle.load(tf)
        pred = new_forest.predict(lung_X)
        assert len(pred) == lung_X.shape[0]

    def test_clone(self, lung_X, lung_y):
        forest = GRFForestSurvival()
        forest.fit(lung_X, lung_y)
        clone(forest)

    def test_equalize_cluster_weights(
        self, lung_X, lung_y, lung_cluster, equalize_cluster_weights
    ):
        forest = GRFForestSurvival(equalize_cluster_weights=equalize_cluster_weights)
        forest.fit(lung_X, lung_y, cluster=lung_cluster)
        if equalize_cluster_weights:
            assert forest.samples_per_cluster_ == 20
        else:
            assert forest.samples_per_cluster_ == lung_y.shape[0] - 20

        if equalize_cluster_weights:
            with pytest.raises(ValueError):
                forest.fit(
                    lung_X,
                    lung_y,
                    cluster=lung_cluster,
                    sample_weight=np.ones(lung_y.shape),
                )

        forest.fit(lung_X, lung_y, cluster=None)
        assert forest.samples_per_cluster_ == 0

    def test_sample_fraction(self, lung_X, lung_y, sample_fraction):
        forest = GRFForestSurvival(sample_fraction=sample_fraction)
        if sample_fraction <= 0 or sample_fraction > 1:
            with pytest.raises(ValueError):
                forest.fit(lung_X, lung_y)
        else:
            forest.fit(lung_X, lung_y)

    def test_mtry(self, lung_X, lung_y, mtry):
        forest = GRFForestSurvival(mtry=mtry)
        forest.fit(lung_X, lung_y)
        if mtry is not None:
            assert forest.mtry_ == mtry
        else:
            assert forest.mtry_ == 3

    def test_honesty(self, lung_X, lung_y, honesty):
        forest = GRFForestSurvival(honesty=honesty)
        forest.fit(lung_X, lung_y)

    def test_honesty_fraction(self, lung_X, lung_y, honesty_fraction):
        forest = GRFForestSurvival(
            honesty=True, honesty_fraction=honesty_fraction, honesty_prune_leaves=True
        )
        if honesty_fraction <= 0 or honesty_fraction >= 1:
            with pytest.raises(RuntimeError):
                forest.fit(lung_X, lung_y)
        else:
            forest.fit(lung_X, lung_y)

    def test_honesty_prune_leaves(self, lung_X, lung_y, honesty_prune_leaves):
        forest = GRFForestSurvival(
            honesty=True, honesty_prune_leaves=honesty_prune_leaves
        )
        forest.fit(lung_X, lung_y)

    def test_alpha(self, lung_X, lung_y, alpha):
        forest = GRFForestSurvival(alpha=alpha)
        if alpha <= 0 or alpha >= 0.25:
            with pytest.raises(ValueError):
                forest.fit(lung_X, lung_y)
        else:
            forest.fit(lung_X, lung_y)

    def test_get_tags(self):
        forest = GRFForestSurvival()
        tags = forest._get_tags()
        assert tags["requires_y"]

    # cant use this because of special fit y
    # def test_check_estimator(self):
    #     check_estimator(GRFSurvival())

    def test_estimators_(self, lung_X, lung_y):
        forest = GRFForestSurvival(n_estimators=10)
        with pytest.raises(AttributeError):
            _ = forest.estimators_
        forest.fit(lung_X, lung_y)
        with pytest.raises(ValueError):
            _ = forest.estimators_
        forest = GRFForestSurvival(n_estimators=10, enable_tree_details=True)
        forest.fit(lung_X, lung_y)
        estimators = forest.estimators_
        assert len(estimators) == 10
        assert isinstance(estimators[0], GRFTreeSurvival)
        check_is_fitted(estimators[0])

    def test_get_estimator(self, lung_X, lung_y):
        forest = GRFForestSurvival(n_estimators=10)
        with pytest.raises(NotFittedError):
            _ = forest.get_estimator(idx=0)
        forest.fit(lung_X, lung_y)
        with pytest.raises(ValueError):
            _ = forest.get_estimator(idx=0)
        forest = GRFForestSurvival(n_estimators=10, enable_tree_details=True)
        forest.fit(lung_X, lung_y)
        estimator = forest.get_estimator(0)
        check_is_fitted(estimator)
        assert isinstance(estimator, GRFTreeSurvival)
        with pytest.raises(IndexError):
            _ = forest.get_estimator(idx=20)

    def test_get_split_frequencies(self, lung_X, lung_y):
        forest = GRFForestSurvival()
        forest.fit(lung_X, lung_y)
        sf = forest.get_split_frequencies()
        assert sf.shape[1] == lung_X.shape[1]

    def test_get_feature_importances(self, lung_X, lung_y):
        forest = GRFForestSurvival()
        forest.fit(lung_X, lung_y)
        fi = forest.get_feature_importances()
        assert len(fi) == lung_X.shape[1]

    def test_get_kernel_weights(self, lung_X, lung_y):
        X_train, X_test, y_train, y_test = train_test_split(
            lung_X, lung_y, test_size=0.33, random_state=42
        )
        forest = GRFForestSurvival()
        forest.fit(X_train, y_train)
        weights = forest.get_kernel_weights(X_test)
        assert weights.shape[0] == X_test.shape[0]
        assert weights.shape[1] == X_train.shape[0]
        oob_weights = forest.get_kernel_weights(X_train, True)
        assert oob_weights.shape[0] == X_train.shape[0]
        assert oob_weights.shape[1] == X_train.shape[0]
