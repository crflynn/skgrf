from contextlib import contextmanager

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree._tree import Tree as SKTree

from skgrf.ensemble import GRFForestCausalRegressor
from skgrf.ensemble import GRFForestClassifier
from skgrf.ensemble import GRFForestInstrumentalRegressor
from skgrf.ensemble import GRFForestLocalLinearRegressor
from skgrf.ensemble import GRFForestQuantileRegressor
from skgrf.ensemble import GRFForestRegressor
from skgrf.tree._tree import Tree


def _getclass(klass):
    """Get a __getattribute__ patch to override __class__."""

    def __class__override(self, item):
        if item == "__class__":
            return klass
        return object.__getattribute__(self, item)

    return __class__override


@contextmanager
def shap_patch():
    """Trick shap into thinking skgrf objects are sklearn objects."""
    tree_orig = Tree.__getattribute__
    Tree.__getattribute__ = _getclass(SKTree)
    regressors = [
        GRFForestRegressor,
        GRFForestLocalLinearRegressor,
        GRFForestCausalRegressor,
        GRFForestInstrumentalRegressor,
        GRFForestQuantileRegressor,
    ]
    originals = []
    for klass in regressors:
        originals.append(klass.__getattribute__)
        klass.__getattribute__ = _getclass(RandomForestRegressor)
    clf_orig = GRFForestClassifier.__getattribute__
    GRFForestClassifier.__getattribute__ = _getclass(RandomForestClassifier)
    yield
    Tree.__getattribute__ = tree_orig
    for klass, orig in zip(regressors, originals):
        klass.__getattribute__ = orig
    GRFForestClassifier.__getattribute__ = clf_orig
