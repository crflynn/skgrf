import pickle
import pytest
import tempfile
from sklearn import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted

from skgrf.ensemble.boosted_regressor import GRFBoostedForestRegressor


class TestGRFBoostedForestRegressor:
    def test_init(self):
        _ = GRFBoostedForestRegressor()

    def test_fit(self, boston_X, boston_y):
        forest = GRFBoostedForestRegressor()
        with pytest.raises(NotFittedError):
            check_is_fitted(forest)
        forest.fit(boston_X, boston_y)
        check_is_fitted(forest)
        assert hasattr(forest, "boosted_forests_")
        assert hasattr(forest, "mtry_")

    def test_predict(self, boston_X, boston_y, boost_predict_steps):
        forest = GRFBoostedForestRegressor()
        forest.fit(boston_X, boston_y)
        pred = forest.predict(boston_X, boost_predict_steps=boost_predict_steps)
        assert len(pred) == boston_X.shape[0]

    def test_serialize(self, boston_X, boston_y):
        forest = GRFBoostedForestRegressor()
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
        forest = GRFBoostedForestRegressor()
        forest.fit(boston_X, boston_y)
        clone(forest)

    def test_equalize_cluster_weights(
        self, boston_X, boston_y, boston_cluster, equalize_cluster_weights
    ):
        forest = GRFBoostedForestRegressor(
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
        forest = GRFBoostedForestRegressor(
            sample_fraction=sample_fraction, ci_group_size=1
        )
        if sample_fraction <= 0 or sample_fraction >= 1:
            with pytest.raises(ValueError):
                forest.fit(boston_X, boston_y)
        else:
            forest.fit(boston_X, boston_y)

        forest = GRFBoostedForestRegressor(
            sample_fraction=sample_fraction, ci_group_size=2
        )
        if sample_fraction <= 0 or sample_fraction > 0.5:
            with pytest.raises(ValueError):
                forest.fit(boston_X, boston_y)
        else:
            forest.fit(boston_X, boston_y)

    def test_mtry(self, boston_X, boston_y, mtry):
        forest = GRFBoostedForestRegressor(mtry=mtry)
        forest.fit(boston_X, boston_y)
        if mtry is not None:
            assert forest.mtry_ == mtry
        else:
            assert forest.mtry_ == 6

    def test_honesty(self, boston_X, boston_y, honesty):
        forest = GRFBoostedForestRegressor(honesty=honesty)
        forest.fit(boston_X, boston_y)

    def test_honesty_fraction(self, boston_X, boston_y, honesty_fraction):
        forest = GRFBoostedForestRegressor(
            honesty=True, honesty_fraction=honesty_fraction, honesty_prune_leaves=True
        )
        if honesty_fraction <= 0 or honesty_fraction >= 1:
            with pytest.raises(RuntimeError):
                forest.fit(boston_X, boston_y)
        else:
            forest.fit(boston_X, boston_y)

    def test_honesty_prune_leaves(self, boston_X, boston_y, honesty_prune_leaves):
        forest = GRFBoostedForestRegressor(
            honesty=True, honesty_prune_leaves=honesty_prune_leaves
        )
        forest.fit(boston_X, boston_y)

    def test_alpha(self, boston_X, boston_y, alpha):
        forest = GRFBoostedForestRegressor(alpha=alpha)
        if alpha <= 0 or alpha >= 0.25:
            with pytest.raises(ValueError):
                forest.fit(boston_X, boston_y)
        else:
            forest.fit(boston_X, boston_y)

    def test_tuning(
        self,
        boston_X,
        boston_y,
        tune_params,
        tune_n_estimators,
        tune_n_reps,
        tune_n_draws,
    ):
        forest = GRFBoostedForestRegressor(
            tune_params=tune_params,
            tune_n_estimators=tune_n_estimators,
            tune_n_reps=tune_n_reps,
            tune_n_draws=tune_n_draws,
        )
        if tune_params == ["invalid"]:
            with pytest.raises(ValueError):
                forest.fit(boston_X, boston_y)
        elif tune_n_draws == 1:
            with pytest.raises(ValueError):
                forest.fit(boston_X, boston_y)
        else:
            forest.fit(boston_X, boston_y)

    def test_boosting(
        self,
        boston_X,
        boston_y,
        boost_steps,
        boost_error_reduction,
        boost_max_steps,
        boost_trees_tune,
    ):
        forest = GRFBoostedForestRegressor(
            tune_params=["mtry"],
            tune_n_draws=5,
            tune_n_reps=2,
            boost_steps=boost_steps,
            boost_error_reduction=boost_error_reduction,
            boost_max_steps=boost_max_steps,
            boost_trees_tune=boost_trees_tune,
        )
        if boost_error_reduction < 0 or boost_error_reduction > 1:
            with pytest.raises(ValueError):
                forest.fit(boston_X, boston_y)
        else:
            forest.fit(boston_X, boston_y)

    def test_check_estimator(self):
        check_estimator(GRFBoostedForestRegressor())
