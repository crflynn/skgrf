from contextlib import contextmanager

from sklearn.tree._tree import Tree as SKTree

from skgrf.tree._tree import Tree


def _getclass(klass):
    """Get a __getattribute__ patch to override __class__."""

    def __class__override(self, item):
        if item == "__class__":
            return klass
        return object.__getattribute__(self, item)

    return __class__override


@contextmanager
def shap_patch(target, using):
    """Trick shap into thinking skgrf objects are sklearn objects."""
    tree_orig = Tree.__getattribute__
    Tree.__getattribute__ = _getclass(SKTree)
    reg_orig = target.__getattribute__
    target.__getattribute__ = _getclass(using)
    yield
    Tree.__getattribute__ = tree_orig
    target.__getattribute__ = reg_orig
