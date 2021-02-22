import numpy as np
import pytest
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sksurv.datasets import load_veterans_lung_cancer

_boston_X, _boston_y = load_boston(return_X_y=True)
_iris_X, _iris_y = load_iris(return_X_y=True)
_lung_X, _lung_y = load_veterans_lung_cancer()


@pytest.fixture
def boston_X():
    return _boston_X


@pytest.fixture
def boston_y():
    return _boston_y


@pytest.fixture
def boston_cluster():
    cluster = np.zeros(_boston_y.shape)
    cluster[20:] = 1
    return cluster


@pytest.fixture
def iris_X():
    return _iris_X


@pytest.fixture
def iris_y():
    return _iris_y


@pytest.fixture
def lung_X():
    # select only the numeric cols
    return _lung_X[["Age_in_years", "Karnofsky_score", "Months_from_Diagnosis"]]


@pytest.fixture
def lung_y():
    return _lung_y


@pytest.fixture
def lung_cluster():
    cluster = np.zeros(_lung_y.shape)
    cluster[20:] = 1
    return cluster


@pytest.fixture(params=[True, False])
def equalize_cluster_weights(request):
    return request.param


@pytest.fixture(params=[-0.1, 0, 0.2, 0.5, 0.8, 1.0, 1.1])
def sample_fraction(request):
    return request.param


@pytest.fixture(params=[2, None])
def mtry(request):
    return request.param


@pytest.fixture(params=[True, False])
def honesty(request):
    return request.param


@pytest.fixture(params=[-0.1, 0, 0.2, 0.5, 0.8, 1.0, 1.1])
def honesty_fraction(request):
    return request.param


@pytest.fixture(params=[True, False])
def honesty_prune_leaves(request):
    return request.param


@pytest.fixture(params=[-0.1, 0, 0.05, 0.1, 0.2, 0.25, 0.3])
def alpha(request):
    return request.param


@pytest.fixture(
    params=[
        ["sample_fraction"],
        ["mtry"],
        ["min_node_size"],
        ["honesty_fraction"],
        ["honesty_prune_leaves"],
        ["alpha"],
        ["imbalance_penalty"],
        ["invalid"],
    ]
)
def tune_params(request):
    return request.param


@pytest.fixture(params=[50])
def tune_n_estimators(request):
    return request.param


@pytest.fixture(params=[2])
def tune_n_reps(request):
    return request.param


@pytest.fixture(params=[1, 5])
def tune_n_draws(request):
    return request.param


@pytest.fixture(params=[2])
def boost_steps(request):
    return request.param


@pytest.fixture(params=[2])
def boost_max_steps(request):
    return request.param


@pytest.fixture(params=[-0.1, 0, 0.5, 1, 1.1])
def boost_error_reduction(request):
    return request.param


@pytest.fixture(params=[10])
def boost_trees_tune(request):
    return request.param


@pytest.fixture(params=[None, 1])
def boost_predict_steps(request):
    return request.param
