import numpy as np
import pickle
import pytest
import tempfile
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from skgrf.ensemble import GRFSurvival


class TestGRFSurvival:
    def test_init(self):
        _ = GRFSurvival()

    def test_fit(self, lung_X, lung_y):
        gfs = GRFSurvival()
        with pytest.raises(NotFittedError):
            check_is_fitted(gfs)
        gfs.fit(lung_X, lung_y)
        check_is_fitted(gfs)
        assert hasattr(gfs, "grf_forest_")
        assert hasattr(gfs, "mtry_")

    def test_predict(self, lung_X, lung_y):
        gfs = GRFSurvival()
        gfs.fit(lung_X, lung_y)
        pred = gfs.predict(lung_X)
        assert len(pred) == lung_X.shape[0]

    def test_serialize(self, lung_X, lung_y):
        gfs = GRFSurvival()
        # not fitted
        tf = tempfile.TemporaryFile()
        pickle.dump(gfs, tf)
        tf.seek(0)
        gfs = pickle.load(tf)
        gfs.fit(lung_X, lung_y)
        # fitted
        tf = tempfile.TemporaryFile()
        pickle.dump(gfs, tf)
        tf.seek(0)
        new_gfs = pickle.load(tf)
        pred = new_gfs.predict(lung_X)
        assert len(pred) == lung_X.shape[0]

    def test_clone(self, lung_X, lung_y):
        gfs = GRFSurvival()
        gfs.fit(lung_X, lung_y)
        clone(gfs)

    def test_equalize_cluster_weights(
        self, lung_X, lung_y, lung_cluster, equalize_cluster_weights
    ):
        gfs = GRFSurvival(equalize_cluster_weights=equalize_cluster_weights)
        gfs.fit(lung_X, lung_y, cluster=lung_cluster)
        if equalize_cluster_weights:
            assert gfs.samples_per_cluster_ == 20
        else:
            assert gfs.samples_per_cluster_ == lung_y.shape[0] - 20

        if equalize_cluster_weights:
            with pytest.raises(ValueError):
                gfs.fit(
                    lung_X,
                    lung_y,
                    cluster=lung_cluster,
                    sample_weight=np.ones(lung_y.shape),
                )

        gfs.fit(lung_X, lung_y, cluster=None)
        assert gfs.samples_per_cluster_ == 0

    def test_sample_fraction(self, lung_X, lung_y, sample_fraction):
        gfs = GRFSurvival(sample_fraction=sample_fraction)
        if sample_fraction <= 0 or sample_fraction > 1:
            with pytest.raises(ValueError):
                gfs.fit(lung_X, lung_y)
        else:
            gfs.fit(lung_X, lung_y)

    def test_mtry(self, lung_X, lung_y, mtry):
        gfs = GRFSurvival(mtry=mtry)
        gfs.fit(lung_X, lung_y)
        if mtry is not None:
            assert gfs.mtry_ == mtry
        else:
            assert gfs.mtry_ == 3

    def test_honesty(self, lung_X, lung_y, honesty):
        gfs = GRFSurvival(honesty=honesty)
        gfs.fit(lung_X, lung_y)

    def test_honesty_fraction(self, lung_X, lung_y, honesty_fraction):
        gfs = GRFSurvival(
            honesty=True, honesty_fraction=honesty_fraction, honesty_prune_leaves=True
        )
        if honesty_fraction <= 0 or honesty_fraction >= 1:
            with pytest.raises(RuntimeError):
                gfs.fit(lung_X, lung_y)
        else:
            gfs.fit(lung_X, lung_y)

    def test_honesty_prune_leaves(self, lung_X, lung_y, honesty_prune_leaves):
        gfs = GRFSurvival(honesty=True, honesty_prune_leaves=honesty_prune_leaves)
        gfs.fit(lung_X, lung_y)

    def test_alpha(self, lung_X, lung_y, alpha):
        gfs = GRFSurvival(alpha=alpha)
        if alpha <= 0 or alpha >= 0.25:
            with pytest.raises(ValueError):
                gfs.fit(lung_X, lung_y)
        else:
            gfs.fit(lung_X, lung_y)

    def test_get_tags(self):
        rfs = GRFSurvival()
        tags = rfs._get_tags()
        assert tags["requires_y"]

    # cant use this because of special fit y
    # def test_check_estimator(self):
    #     check_estimator(GRFSurvival())
