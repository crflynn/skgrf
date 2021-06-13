import numpy as np
from sklearn.base import BaseEstimator
from sklearn.tree._tree import csr_matrix
from sklearn.utils.validation import check_is_fitted

from skgrf.base import GRFMixin


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


class Tree:
    """The low-level tree interface.

    Tree objects can be accessed using the ``tree_`` attribute on fitted
    GRF decision tree estimators. Instances of ``Tree`` provide methods and
    properties describing the underlying structure and attributes of the
    tree.
    """

    def __init__(self, grf_forest):
        self.grf_forest = grf_forest

    def get_depth(self):
        """Calculate the maximum depth of the tree."""
        left = self.grf_forest["child_nodes"][0][0]
        right = self.grf_forest["child_nodes"][0][1]
        root_node = self._root_node_index
        return self._get_depth(left, right, root_node)

    def _get_depth(self, left, right, idx):
        if left[idx] == 0:
            return 0
        return 1 + max(
            self._get_depth(left, right, left[idx]),
            self._get_depth(left, right, right[idx]),
        )

    def get_n_leaves(self):
        """Calculate the number of leaves of the tree."""
        left = self.grf_forest["child_nodes"][0][0]
        right = self.grf_forest["child_nodes"][0][1]
        root_node = self._root_node_index
        return self._get_n_leaves(left, right, root_node)

    def _get_n_leaves(self, left, right, idx):
        if left[idx] == 0:
            return 1
        return self._get_n_leaves(left, right, left[idx]) + self._get_n_leaves(
            left, right, right[idx]
        )

    def apply(self, X):
        """Calculate the leaf index for each sample.

        :param array2d X: training input features
        """
        return np.apply_along_axis(self._apply, 1, X)

    def _apply(self, x, idx=None):
        if idx is None:
            idx = self._root_node_index
            return self._apply(x, idx)
        if self.grf_forest["child_nodes"][0][0][idx] == 0:
            return idx
        varid = self.grf_forest["split_vars"][0][idx]
        val = self.grf_forest["split_vars"][0][idx]
        x_val = x[varid]
        if np.isnan(x_val) or x_val is None:
            if self.grf_forest["send_missing_left"][0][idx]:
                idx = self.grf_forest["child_nodes"][0][0][idx]
            else:
                idx = self.grf_forest["child_nodes"][0][1][idx]
        else:
            if x[varid] <= val:
                idx = self.grf_forest["child_nodes"][0][0][idx]
            else:
                idx = self.grf_forest["child_nodes"][0][1][idx]
        return self._apply(x, idx)

    def decision_path(self, X):
        """Calculate the decision path through the tree for each sample.

        :param array2d X: training input features
        """
        if hasattr(X, "values"):  # pd.Dataframe
            Xvalues = X.values
        else:
            Xvalues = X
        paths = [self._decision_path(x) for x in Xvalues]
        rows = [np.ones(len(p), dtype=int) * idx for idx, p in enumerate(paths)]
        rows = np.concatenate(rows, axis=0)
        cols = np.concatenate(paths, axis=0)
        data = np.ones(len(rows), dtype=int)
        return csr_matrix((data, (rows, cols)))

    def _decision_path(self, x, idx=None):
        if idx is None:
            idx = self._root_node_index
            return [idx] + self._decision_path(x, idx)
        if self.grf_forest["child_nodes"][0][0][idx] == 0:
            return []
        varid = self.grf_forest["split_vars"][0][idx]
        val = self.grf_forest["split_values"][0][idx]
        x_val = x[varid]
        if np.isnan(x_val) or x_val is None:
            if self.grf_forest["send_missing_left"][0][idx]:
                idx = self.grf_forest["child_nodes"][0][0][idx]
            else:
                idx = self.grf_forest["child_nodes"][0][1][idx]
        else:
            if x[varid] <= val:
                idx = self.grf_forest["child_nodes"][0][0][idx]
            else:
                idx = self.grf_forest["child_nodes"][0][1][idx]
        return [idx] + self._decision_path(x, idx)

    @property
    def max_depth(self):
        """Max depth of the tree."""
        return self.get_depth()

    @property
    def children_left(self):
        """Left children nodes."""
        # sklearn uses -1, grf uses 0
        return np.array(
            [-1 if n == 0 else n for n in self.grf_forest["child_nodes"][0][0]]
        )

    @property
    def children_right(self):
        """Right children nodes."""
        # sklearn uses -1, grf uses 0
        return np.array(
            [-1 if n == 0 else n for n in self.grf_forest["child_nodes"][0][1]]
        )

    @property
    def children_default(self):
        """Children nodes for missing data."""
        children_left = self.children_left
        children_right = self.children_right
        children_default = [
            children_left[idx] if val else children_right[idx]
            for idx, val in enumerate(self.grf_forest["send_missing_left"][0])
        ]
        return np.array(children_default)

    @property
    def feature(self):
        """Variables on which nodes are split."""
        return np.array(self.grf_forest["split_vars"][0])

    @property
    def threshold(self):
        """Threshold values on which nodes are split."""
        return np.array(self.grf_forest["split_values"][0])

    @property
    def n_node_samples(self):
        """The number of samples reaching each node."""
        n_samples = [len(node) for node in self.grf_forest["leaf_samples"][0]]
        self._get_n_node_samples(
            self.children_left, self.children_right, self._root_node_index, n_samples
        )
        return np.array(n_samples)

    @property
    def _root_node_index(self):
        return self.grf_forest["root_nodes"][0]

    def _get_n_node_samples(self, left, right, idx, n_samples):
        left_n_node_samples = (
            n_samples[idx]
            if left[idx] == -1
            else self._get_n_node_samples(left, right, left[idx], n_samples)
        )
        right_n_node_samples = (
            n_samples[idx]
            if right[idx] == -1
            else self._get_n_node_samples(left, right, right[idx], n_samples)
        )
        n_samples[idx] = left_n_node_samples + right_n_node_samples
        return n_samples[idx]

    @property
    def weighted_n_node_samples(self):
        """The sum of the weights of the samples reaching each node."""
        weighted_n_samples = self.grf_forest["leaf_weights"][0].copy()
        self._get_n_node_samples(
            self.children_left,
            self.children_right,
            self._root_node_index,
            weighted_n_samples,
        )
        return np.array(weighted_n_samples)
