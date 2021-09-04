import numpy as np
import pickle
import pytest
import tempfile
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.tree._tree import csr_matrix
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted

from skgrf.ensemble import GRFForestClassifier
from skgrf.tree.classifier import GRFTreeClassifier


class TestGRFTreeClassifier:
    def test_init(self):
        _ = GRFTreeClassifier()

    def test_fit(self, iris_X, iris_y):
        tree = GRFTreeClassifier()
        with pytest.raises(NotFittedError):
            check_is_fitted(tree)
        tree.fit(iris_X, iris_y)
        check_is_fitted(tree)
        assert hasattr(tree, "grf_forest_")
        assert hasattr(tree, "mtry_")
        assert tree.grf_forest_["num_trees"] == 1
        assert tree.criterion == "gini"

    def test_predict(self, iris_X, iris_y):
        tree = GRFTreeClassifier()
        tree.fit(iris_X, iris_y)
        pred = tree.predict(iris_X)
        assert len(pred) == iris_X.shape[0]

    def test_with_X_nan(self, iris_X, iris_y):
        iris_X_nan = iris_X.copy()
        index = np.random.choice(iris_X_nan.size, 100, replace=False)
        iris_X_nan.ravel()[index] = np.nan
        assert np.sum(np.isnan(iris_X_nan)) == 100
        tree = GRFTreeClassifier()
        tree.fit(iris_X_nan, iris_y)
        pred = tree.predict(iris_X_nan)
        assert len(pred) == iris_X_nan.shape[0]

    def test_predict_oob(self, iris_X, iris_y):
        tree = GRFTreeClassifier()
        tree.fit(iris_X, iris_y, compute_oob_predictions=True)
        pred = np.atleast_1d(np.squeeze(np.array(tree.grf_forest_["predictions"])))
        assert len(pred) == iris_X.shape[0]

    def test_predict_proba(self, iris_X, iris_y):
        tree = GRFTreeClassifier()
        tree.fit(iris_X, iris_y)
        pred_proba = tree.predict_proba(iris_X)
        assert pred_proba.shape == (iris_X.shape[0], tree.n_classes_)

    def test_predict_log_proba(self, iris_X, iris_y):
        tree = GRFTreeClassifier()
        tree.fit(iris_X, iris_y)
        pred_log_proba = tree.predict_log_proba(iris_X)
        assert pred_log_proba.shape == (iris_X.shape[0], tree.n_classes_)

    def test_serialize(self, iris_X, iris_y):
        tree = GRFTreeClassifier()
        # not fitted
        tf = tempfile.TemporaryFile()
        pickle.dump(tree, tf)
        tf.seek(0)
        tree = pickle.load(tf)
        tree.fit(iris_X, iris_y)
        # fitted
        tf = tempfile.TemporaryFile()
        pickle.dump(tree, tf)
        tf.seek(0)
        new_tree = pickle.load(tf)
        pred = new_tree.predict(iris_X)
        assert len(pred) == iris_X.shape[0]

    def test_clone(self, iris_X, iris_y):
        tree = GRFTreeClassifier()
        tree.fit(iris_X, iris_y)
        clone(tree)

    def test_equalize_cluster_weights(
        self, iris_X, iris_y, iris_cluster, equalize_cluster_weights
    ):
        tree = GRFTreeClassifier(equalize_cluster_weights=equalize_cluster_weights)
        tree.fit(iris_X, iris_y, cluster=iris_cluster)
        if equalize_cluster_weights:
            assert tree.samples_per_cluster_ == 20
        else:
            assert tree.samples_per_cluster_ == iris_y.shape[0] - 20

        if equalize_cluster_weights:
            with pytest.raises(ValueError):
                tree.fit(iris_X, iris_y, cluster=iris_cluster, sample_weight=iris_y)

        tree.fit(iris_X, iris_y, cluster=None)
        assert tree.samples_per_cluster_ == 0

    def test_sample_fraction(
        self, iris_X, iris_y, sample_fraction
    ):  # and ci_group_size
        tree = GRFTreeClassifier(sample_fraction=sample_fraction)
        if sample_fraction <= 0 or sample_fraction > 1:
            with pytest.raises(ValueError):
                tree.fit(iris_X, iris_y)
        else:
            tree.fit(iris_X, iris_y)

    def test_mtry(self, iris_X, iris_y, mtry):
        tree = GRFTreeClassifier(mtry=mtry)
        tree.fit(iris_X, iris_y)
        if mtry is not None:
            assert tree.mtry_ == mtry
        else:
            assert tree.mtry_ == 4

    def test_honesty(self, iris_X, iris_y, honesty):
        tree = GRFTreeClassifier(honesty=honesty)
        tree.fit(iris_X, iris_y)

    def test_honesty_fraction(self, iris_X, iris_y, honesty_fraction):
        tree = GRFTreeClassifier(
            honesty=True, honesty_fraction=honesty_fraction, honesty_prune_leaves=True
        )
        if honesty_fraction <= 0 or honesty_fraction >= 1:
            with pytest.raises(RuntimeError):
                tree.fit(iris_X, iris_y)
        else:
            tree.fit(iris_X, iris_y)

    def test_honesty_prune_leaves(self, iris_X, iris_y, honesty_prune_leaves):
        tree = GRFTreeClassifier(
            honesty=True, honesty_prune_leaves=honesty_prune_leaves
        )
        tree.fit(iris_X, iris_y)

    def test_alpha(self, iris_X, iris_y, alpha):
        tree = GRFTreeClassifier(alpha=alpha)
        if alpha <= 0 or alpha >= 0.25:
            with pytest.raises(ValueError):
                tree.fit(iris_X, iris_y)
        else:
            tree.fit(iris_X, iris_y)

    def test_check_estimator(self):
        # using honesty here means that the test
        # `check_classifiers_predictions` will fail, because
        # the test dataset is very small. the failure occurs
        # when comparing y == y_pred on binary classification
        check_estimator(GRFTreeClassifier(honesty=False))

        with pytest.raises(AssertionError) as exc:
            check_estimator(GRFTreeClassifier(honesty=True))
            assert "Arrays are not equal" in exc

    def test_from_forest(self, iris_X, iris_y):
        forest = GRFForestClassifier()
        forest.fit(iris_X, iris_y)
        tree = GRFTreeClassifier.from_forest(forest=forest, idx=0)
        tree.predict(iris_X)

    # region base
    def test_get_depth(self, iris_X, iris_y):
        tree = GRFTreeClassifier()
        tree.fit(iris_X, iris_y)
        depth = tree.get_depth()
        assert isinstance(depth, int)
        assert depth > 0

    def test_get_n_leaves(self, iris_X, iris_y):
        tree = GRFTreeClassifier()
        tree.fit(iris_X, iris_y)
        leaves = tree.get_n_leaves()
        assert isinstance(leaves, int)
        assert np.all(leaves > 0)

    def test_apply(self, iris_X, iris_y):
        tree = GRFTreeClassifier()
        tree.fit(iris_X, iris_y)
        leaves = tree.apply(iris_X)
        assert isinstance(leaves, np.ndarray)
        assert np.all(leaves > 0)
        assert len(leaves) == len(iris_X)

    def test_decision_path(self, iris_X, iris_y):
        tree = GRFTreeClassifier()
        tree.fit(iris_X, iris_y)
        paths = tree.decision_path(iris_X)
        assert isinstance(paths, csr_matrix)
        assert paths.shape[0] == len(iris_X)

    def test_tree_interface(self, iris_X, iris_y):
        tree = GRFTreeClassifier()
        tree.fit(iris_X, iris_y)
        # access attributes the way we would expect to in sklearn
        tree_ = tree.tree_
        children_left = tree_.children_left
        children_right = tree_.children_right
        children_default = tree_.children_default
        feature = tree_.feature
        threshold = tree_.threshold
        max_depth = tree_.max_depth
        n_node_samples = tree_.n_node_samples
        weighted_n_node_samples = tree_.weighted_n_node_samples
        node_count = tree_.node_count
        capacity = tree_.capacity
        n_outputs = tree_.n_outputs
        n_classes = tree_.n_classes
        value = tree_.value

    # endregion
