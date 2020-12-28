import pytest
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris

_boston_X, _boston_y = load_boston(return_X_y=True)
_iris_X, _iris_y = load_iris(return_X_y=True)


@pytest.fixture
def boston_X():
    return _boston_X


@pytest.fixture
def boston_y():
    return _boston_y


@pytest.fixture
def iris_X():
    return _iris_X


@pytest.fixture
def iris_y():
    return _iris_y
