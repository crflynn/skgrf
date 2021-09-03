import numpy as np
import pickle
import pytest
import tempfile
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted

from skgrf.ensemble import GRFForestClassifier
from skgrf.tree import GRFTreeClassifier


class TestGRFForestClassifier:
    def test_init(self):
        _ = GRFForestClassifier()

    def test_fit(self, iris_X, iris_y):
        forest = GRFForestClassifier()
        with pytest.raises(NotFittedError):
            check_is_fitted(forest)
        forest.fit(iris_X, iris_y)
        check_is_fitted(forest)
        assert hasattr(forest, "grf_forest_")
        assert hasattr(forest, "mtry_")
        assert forest.criterion == "gini"

    def test_predict(self, iris_X, iris_y):
        forest = GRFForestClassifier()
        forest.fit(iris_X, iris_y)
        pred = forest.predict(iris_X)
        assert len(pred) == iris_X.shape[0]

    def test_with_X_nan(self, iris_X, iris_y):
        index = np.random.choice(iris_X.size, 100, replace=False)
        iris_X.ravel()[index] = np.nan
        forest = GRFForestClassifier()
        forest.fit(iris_X, iris_y)
        pred = forest.predict(iris_X)
        assert len(pred) == iris_X.shape[0]

    def test_predict_oob(self, iris_X, iris_y):
        forest = GRFForestClassifier()
        forest.fit(iris_X, iris_y, compute_oob_predictions=True)
        pred = np.atleast_1d(np.squeeze(np.array(forest.grf_forest_["predictions"])))
        assert len(pred) == iris_X.shape[0]

    def test_predict_proba(self, iris_X, iris_y):
        forest = GRFForestClassifier()
        forest.fit(iris_X, iris_y)
        pred_proba = forest.predict_proba(iris_X)
        assert pred_proba.shape == (iris_X.shape[0], forest.n_classes_)

    def test_predict_log_proba(self, iris_X, iris_y):
        forest = GRFForestClassifier()
        forest.fit(iris_X, iris_y)
        pred_log_proba = forest.predict_log_proba(iris_X)
        assert pred_log_proba.shape == (iris_X.shape[0], forest.n_classes_)

    def test_serialize(self, iris_X, iris_y):
        forest = GRFForestClassifier()
        # not fitted
        tf = tempfile.TemporaryFile()
        pickle.dump(forest, tf)
        tf.seek(0)
        forest = pickle.load(tf)
        forest.fit(iris_X, iris_y)
        # fitted
        tf = tempfile.TemporaryFile()
        pickle.dump(forest, tf)
        tf.seek(0)
        new_forest = pickle.load(tf)
        pred = new_forest.predict(iris_X)
        assert len(pred) == iris_X.shape[0]

    def test_clone(self, iris_X, iris_y):
        forest = GRFForestClassifier()
        forest.fit(iris_X, iris_y)
        clone(forest)

    def test_equalize_cluster_weights(
        self, iris_X, iris_y, iris_cluster, equalize_cluster_weights
    ):
        forest = GRFForestClassifier(equalize_cluster_weights=equalize_cluster_weights)
        forest.fit(iris_X, iris_y, cluster=iris_cluster)
        if equalize_cluster_weights:
            assert forest.samples_per_cluster_ == 20
        else:
            assert forest.samples_per_cluster_ == iris_y.shape[0] - 20

        if equalize_cluster_weights:
            with pytest.raises(ValueError):
                forest.fit(iris_X, iris_y, cluster=iris_cluster, sample_weight=iris_y)

        forest.fit(iris_X, iris_y, cluster=None)
        assert forest.samples_per_cluster_ == 0

    def test_sample_fraction(
        self, iris_X, iris_y, sample_fraction
    ):  # and ci_group_size
        forest = GRFForestClassifier(sample_fraction=sample_fraction, ci_group_size=1)
        if sample_fraction <= 0 or sample_fraction > 1:
            with pytest.raises(ValueError):
                forest.fit(iris_X, iris_y)
        else:
            forest.fit(iris_X, iris_y)

        forest = GRFForestClassifier(sample_fraction=sample_fraction, ci_group_size=2)
        if sample_fraction <= 0 or sample_fraction > 0.5:
            with pytest.raises(ValueError):
                forest.fit(iris_X, iris_y)
        else:
            forest.fit(iris_X, iris_y)

    def test_mtry(self, iris_X, iris_y, mtry):
        forest = GRFForestClassifier(mtry=mtry)
        forest.fit(iris_X, iris_y)
        if mtry is not None:
            assert forest.mtry_ == mtry
        else:
            assert forest.mtry_ == 4

    def test_honesty(self, iris_X, iris_y, honesty):
        forest = GRFForestClassifier(honesty=honesty)
        forest.fit(iris_X, iris_y)

    def test_honesty_fraction(self, iris_X, iris_y, honesty_fraction):
        forest = GRFForestClassifier(
            honesty=True, honesty_fraction=honesty_fraction, honesty_prune_leaves=True
        )
        if honesty_fraction <= 0 or honesty_fraction >= 1:
            with pytest.raises(RuntimeError):
                forest.fit(iris_X, iris_y)
        else:
            forest.fit(iris_X, iris_y)

    def test_honesty_prune_leaves(self, iris_X, iris_y, honesty_prune_leaves):
        forest = GRFForestClassifier(
            honesty=True, honesty_prune_leaves=honesty_prune_leaves
        )
        forest.fit(iris_X, iris_y)

    def test_alpha(self, iris_X, iris_y, alpha):
        forest = GRFForestClassifier(alpha=alpha)
        if alpha <= 0 or alpha >= 0.25:
            with pytest.raises(ValueError):
                forest.fit(iris_X, iris_y)
        else:
            forest.fit(iris_X, iris_y)

    def test_check_estimator(self):
        # using honesty here means that the test
        # `check_classifiers_predictions` will fail, because
        # the test dataset is very small. the failure occurs
        # when comparing y == y_pred on binary classification
        check_estimator(GRFForestClassifier(honesty=False))

        with pytest.raises(AssertionError) as exc:
            check_estimator(GRFForestClassifier(honesty=True))
            assert "Arrays are not equal" in exc

    def test_estimators_(self, iris_X, iris_y):
        forest = GRFForestClassifier(n_estimators=10)
        with pytest.raises(AttributeError):
            _ = forest.estimators_
        forest.fit(iris_X, iris_y)
        with pytest.raises(ValueError):
            _ = forest.estimators_
        forest = GRFForestClassifier(n_estimators=10, enable_tree_details=True)
        forest.fit(iris_X, iris_y)
        estimators = forest.estimators_
        assert len(estimators) == 10
        assert isinstance(estimators[0], GRFTreeClassifier)
        check_is_fitted(estimators[0])

    def test_get_estimator(self, iris_X, iris_y):
        forest = GRFForestClassifier(n_estimators=10)
        with pytest.raises(NotFittedError):
            _ = forest.get_estimator(idx=0)
        forest.fit(iris_X, iris_y)
        with pytest.raises(ValueError):
            _ = forest.get_estimator(idx=0)
        forest = GRFForestClassifier(n_estimators=10, enable_tree_details=True)
        forest.fit(iris_X, iris_y)
        estimator = forest.get_estimator(idx=0)
        check_is_fitted(estimator)
        assert isinstance(estimator, GRFTreeClassifier)
        with pytest.raises(IndexError):
            _ = forest.get_estimator(idx=20)

    def test_get_split_frequencies(self, iris_X, iris_y):
        forest = GRFForestClassifier()
        forest.fit(iris_X, iris_y)
        sf = forest.get_split_frequencies()
        assert sf.shape[1] == iris_X.shape[1]

    def test_get_feature_importances(self, iris_X, iris_y):
        forest = GRFForestClassifier()
        forest.fit(iris_X, iris_y)
        fi = forest.get_feature_importances()
        assert len(fi) == iris_X.shape[1]

    def test_get_kernel_weights(self, iris_X, iris_y):
        X_train, X_test, y_train, y_test = train_test_split(
            iris_X, iris_y, test_size=0.33, random_state=42
        )
        forest = GRFForestClassifier()
        forest.fit(X_train, y_train)
        weights = forest.get_kernel_weights(X_test)
        assert weights.shape[0] == X_test.shape[0]
        assert weights.shape[1] == X_train.shape[0]
        oob_weights = forest.get_kernel_weights(X_train, True)
        assert oob_weights.shape[0] == X_train.shape[0]
        assert oob_weights.shape[1] == X_train.shape[0]

    def test_accuracy(self, iris_X, iris_y):
        X_train, X_test, y_train, y_test = train_test_split(
            iris_X, iris_y, test_size=0.33, random_state=42
        )

        # train and test a random forest classifier
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        rf_acc = accuracy_score(y_test, y_pred_rf)

        # train and test a ranger classifier
        ra = GRFForestClassifier()
        ra.fit(X_train, y_train)
        y_pred_ra = ra.predict(X_test)
        ranger_acc = accuracy_score(y_test, y_pred_ra)

        # the accuracy should be good
        assert rf_acc > 0.9
        assert ranger_acc > 0.9
