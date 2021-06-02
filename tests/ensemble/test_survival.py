import numpy as np
import pickle
import pytest
import tempfile
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from skgrf.ensemble.survival import GRFSurvival
from skgrf.tree.survival import GRFTreeSurvival


class TestGRFSurvival:
    def test_init(self):
        _ = GRFSurvival()

    def test_fit(self, lung_X, lung_y):
        forest = GRFSurvival()
        with pytest.raises(NotFittedError):
            check_is_fitted(forest)
        forest.fit(lung_X, lung_y)
        check_is_fitted(forest)
        assert hasattr(forest, "grf_forest_")
        assert hasattr(forest, "mtry_")

    def test_predict(self, lung_X, lung_y):
        forest = GRFSurvival()
        forest.fit(lung_X, lung_y)
        pred = forest.predict(lung_X)
        assert len(pred) == lung_X.shape[0]

    def test_serialize(self, lung_X, lung_y):
        forest = GRFSurvival()
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
        forest = GRFSurvival()
        forest.fit(lung_X, lung_y)
        clone(forest)

    def test_equalize_cluster_weights(
        self, lung_X, lung_y, lung_cluster, equalize_cluster_weights
    ):
        forest = GRFSurvival(equalize_cluster_weights=equalize_cluster_weights)
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
        forest = GRFSurvival(sample_fraction=sample_fraction)
        if sample_fraction <= 0 or sample_fraction > 1:
            with pytest.raises(ValueError):
                forest.fit(lung_X, lung_y)
        else:
            forest.fit(lung_X, lung_y)

    def test_mtry(self, lung_X, lung_y, mtry):
        forest = GRFSurvival(mtry=mtry)
        forest.fit(lung_X, lung_y)
        if mtry is not None:
            assert forest.mtry_ == mtry
        else:
            assert forest.mtry_ == 3

    def test_honesty(self, lung_X, lung_y, honesty):
        forest = GRFSurvival(honesty=honesty)
        forest.fit(lung_X, lung_y)

    def test_honesty_fraction(self, lung_X, lung_y, honesty_fraction):
        forest = GRFSurvival(
            honesty=True, honesty_fraction=honesty_fraction, honesty_prune_leaves=True
        )
        if honesty_fraction <= 0 or honesty_fraction >= 1:
            with pytest.raises(RuntimeError):
                forest.fit(lung_X, lung_y)
        else:
            forest.fit(lung_X, lung_y)

    def test_honesty_prune_leaves(self, lung_X, lung_y, honesty_prune_leaves):
        forest = GRFSurvival(honesty=True, honesty_prune_leaves=honesty_prune_leaves)
        forest.fit(lung_X, lung_y)

    def test_alpha(self, lung_X, lung_y, alpha):
        forest = GRFSurvival(alpha=alpha)
        if alpha <= 0 or alpha >= 0.25:
            with pytest.raises(ValueError):
                forest.fit(lung_X, lung_y)
        else:
            forest.fit(lung_X, lung_y)

    def test_get_tags(self):
        forest = GRFSurvival()
        tags = forest._get_tags()
        assert tags["requires_y"]

    # cant use this because of special fit y
    # def test_check_estimator(self):
    #     check_estimator(GRFSurvival())

    def test_estimators_(self, lung_X, lung_y):
        forest = GRFSurvival(n_estimators=10)
        with pytest.raises(AttributeError):
            _ = forest.estimators_
        forest.fit(lung_X, lung_y)
        estimators = forest.estimators_
        assert len(estimators) == 10
        assert isinstance(estimators[0], GRFTreeSurvival)
        check_is_fitted(estimators[0])
