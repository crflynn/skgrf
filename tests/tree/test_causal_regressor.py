import pickle
import tempfile

import numpy as np
import pytest
from sklearn import clone
from sklearn.exceptions import NotFittedError
from sklearn.tree._tree import csr_matrix
from sklearn.utils.validation import check_is_fitted

from skgrf.ensemble import GRFForestCausalRegressor
from skgrf.tree import GRFTreeCausalRegressor


class TestGRFTreeCausalRegressor:
    def test_init(self):
        _ = GRFTreeCausalRegressor()

    def test_fit(self, causal_X, causal_y, causal_w):
        tree = GRFTreeCausalRegressor()
        with pytest.raises(NotFittedError):
            check_is_fitted(tree)
        tree.fit(causal_X, causal_y, causal_w)
        check_is_fitted(tree)
        assert hasattr(tree, "grf_forest_")
        assert hasattr(tree, "mtry_")
        assert tree.grf_forest_["num_trees"] == 1
        assert tree.criterion == "mse"

    def test_predict(self, causal_X, causal_y, causal_w):
        tree = GRFTreeCausalRegressor()
        tree.fit(causal_X, causal_y, causal_w)
        pred = tree.predict(causal_X)
        assert len(pred) == causal_X.shape[0]

    def test_with_X_nan(self, causal_X, causal_y, causal_w):
        causal_X_nan = causal_X.copy()
        index = np.random.choice(causal_X_nan.size, 100, replace=False)
        causal_X_nan.ravel()[index] = np.nan
        assert np.sum(np.isnan(causal_X_nan)) == 100
        tree = GRFTreeCausalRegressor()
        tree.fit(causal_X_nan, causal_y, causal_w)
        pred = tree.predict(causal_X_nan)
        assert len(pred) == causal_X_nan.shape[0]

    def test_serialize(self, causal_X, causal_y, causal_w):
        tree = GRFTreeCausalRegressor()
        # not fitted
        tf = tempfile.TemporaryFile()
        pickle.dump(tree, tf)
        tf.seek(0)
        tree = pickle.load(tf)
        tree.fit(causal_X, causal_y, causal_w)
        # fitted
        tf = tempfile.TemporaryFile()
        pickle.dump(tree, tf)
        tf.seek(0)
        new_tree = pickle.load(tf)
        pred = new_tree.predict(causal_X)
        assert len(pred) == causal_X.shape[0]

    def test_clone(self, causal_X, causal_y, causal_w):
        tree = GRFTreeCausalRegressor()
        tree.fit(causal_X, causal_y, causal_w)
        clone(tree)

    def test_equalize_cluster_weights(
        self,
        causal_X,
        causal_y,
        causal_w,
        causal_cluster,
        equalize_cluster_weights,
    ):
        gcr = GRFTreeCausalRegressor(equalize_cluster_weights=equalize_cluster_weights)
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
        gcr = GRFTreeCausalRegressor(sample_fraction=sample_fraction)
        if sample_fraction <= 0 or sample_fraction > 1:
            with pytest.raises(ValueError):
                gcr.fit(causal_X, causal_y, causal_w, causal_w)
        else:
            gcr.fit(causal_X, causal_y, causal_w, causal_w)

    def test_mtry(self, causal_X, causal_y, causal_w, mtry):
        gcr = GRFTreeCausalRegressor(mtry=mtry)
        gcr.fit(causal_X, causal_y, causal_w, causal_w)
        if mtry is not None:
            assert gcr.mtry_ == mtry
        else:
            assert gcr.mtry_ == 5

    def test_honesty(self, causal_X, causal_y, causal_w, honesty):
        gcr = GRFTreeCausalRegressor(honesty=honesty)
        gcr.fit(causal_X, causal_y, causal_w, causal_w)

    def test_honesty_fraction(self, causal_X, causal_y, causal_w, honesty_fraction):
        gcr = GRFTreeCausalRegressor(
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
        gcr = GRFTreeCausalRegressor(
            honesty=True, honesty_prune_leaves=honesty_prune_leaves
        )
        gcr.fit(causal_X, causal_y, causal_w, causal_w)

    def test_alpha(self, causal_X, causal_y, causal_w, alpha):
        gcr = GRFTreeCausalRegressor(alpha=alpha)
        if alpha <= 0 or alpha >= 0.25:
            with pytest.raises(ValueError):
                gcr.fit(causal_X, causal_y, causal_w, causal_w)
        else:
            gcr.fit(causal_X, causal_y, causal_w, causal_w)

    def test_orthogonal_boosting(
        self, causal_X, causal_y, causal_w, orthogonal_boosting
    ):
        gcr = GRFTreeCausalRegressor(orthogonal_boosting=orthogonal_boosting)
        gcr.fit(causal_X, causal_y, causal_w)

    # cant use this because of extra required fit params
    # def test_check_estimator(self):
    #     check_estimator(GRFTreeCausalRegressor())

    def test_from_forest(self, causal_X, causal_y, causal_w):
        forest = GRFForestCausalRegressor()
        forest.fit(causal_X, causal_y, causal_w)
        tree = GRFTreeCausalRegressor.from_forest(forest=forest, idx=0)
        tree.predict(causal_X)

    # region base
    def test_get_depth(self, causal_X, causal_y, causal_w):
        tree = GRFTreeCausalRegressor()
        tree.fit(causal_X, causal_y, causal_w)
        depth = tree.get_depth()
        assert isinstance(depth, int)
        assert depth > 0

    def test_get_n_leaves(self, causal_X, causal_y, causal_w):
        tree = GRFTreeCausalRegressor()
        tree.fit(causal_X, causal_y, causal_w)
        leaves = tree.get_n_leaves()
        assert isinstance(leaves, int)
        assert leaves > 0

    def test_apply(self, causal_X, causal_y, causal_w):
        tree = GRFTreeCausalRegressor()
        tree.fit(causal_X, causal_y, causal_w)
        leaves = tree.apply(causal_X)
        assert isinstance(leaves, np.ndarray)
        assert np.all(leaves > 0)
        assert len(leaves) == len(causal_X)

    def test_decision_path(self, causal_X, causal_y, causal_w):
        tree = GRFTreeCausalRegressor()
        tree.fit(causal_X, causal_y, causal_w)
        paths = tree.decision_path(causal_X)
        assert isinstance(paths, csr_matrix)
        assert paths.shape[0] == len(causal_X)

    def test_tree_interface(self, causal_X, causal_y, causal_w):
        tree = GRFTreeCausalRegressor()
        tree.fit(causal_X, causal_y, causal_w)
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
