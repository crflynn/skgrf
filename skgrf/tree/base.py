from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from skgrf.base import GRFMixin
from skgrf.tree._tree import Tree


class BaseGRFTree(GRFMixin, BaseEstimator):
    @property
    def tree_(self):
        check_is_fitted(self)
        return Tree(grf_forest=self.grf_forest_)

    def get_depth(self):
        """Calculate the maximum depth of the tree."""
        return self.tree_.get_depth()

    def get_n_leaves(self):
        """Calculate the number of leaves of the tree."""
        return self.tree_.get_n_leaves()

    def apply(self, X):
        """Calculate the index of the leaf for each sample.

        :param array2d X: training input features
        """
        return self.tree_.apply(X)

    def decision_path(self, X):
        """Calculate the decision path through the tree for each sample.

        :param array2d X: training input features
        """
        return self.tree_.decision_path(X)
