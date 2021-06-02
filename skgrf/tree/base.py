import numpy as np
from sklearn.tree._tree import csr_matrix


class BaseGRFTree:
    def get_depth(self):
        """Calculate the maximum depth of the tree."""
        left = self.grf_forest_["_child_nodes"][0][0]
        right = self.grf_forest_["_child_nodes"][0][1]
        root_node = self.grf_forest_["_root_nodes"][0]
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
        left = self.grf_forest_["_child_nodes"][0][0]
        right = self.grf_forest_["_child_nodes"][0][1]
        root_node = self.grf_forest_["_root_nodes"][0]
        return self._get_n_leaves(left, right, root_node)

    def _get_n_leaves(self, left, right, idx):
        if left[idx] == 0:
            return 1
        return self._get_n_leaves(left, right, left[idx]) + self._get_n_leaves(
            left, right, right[idx]
        )

    def apply(self, X):
        """Calculate the index of the leaf for each sample.

        :param array2d X: training input features
        """
        return np.apply_along_axis(self._apply, 1, X)

    def _apply(self, x, idx=None):
        if idx is None:
            idx = self.grf_forest_["_root_nodes"][0]
            return self._apply(x, idx)
        if self.grf_forest_["_child_nodes"][0][0][idx] == 0:
            return idx
        varid = self.grf_forest_["_split_vars"][0][idx]
        val = self.grf_forest_["_split_vars"][0][idx]
        x_val = x[varid]
        if np.isnan(x_val) or x_val is None:
            if self.grf_forest_["_send_missing_left"][0][idx]:
                idx = self.grf_forest_["_child_nodes"][0][0][idx]
            else:
                idx = self.grf_forest_["_child_nodes"][0][1][idx]
        else:
            if x[varid] <= val:
                idx = self.grf_forest_["_child_nodes"][0][0][idx]
            else:
                idx = self.grf_forest_["_child_nodes"][0][1][idx]
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
            idx = self.grf_forest_["_root_nodes"][0]
            return [idx] + self._decision_path(x, idx)
        if self.grf_forest_["_child_nodes"][0][0][idx] == 0:
            return []
        varid = self.grf_forest_["_split_vars"][0][idx]
        val = self.grf_forest_["_split_vars"][0][idx]
        x_val = x[varid]
        if np.isnan(x_val) or x_val is None:
            if self.grf_forest_["_send_missing_left"][0][idx]:
                idx = self.grf_forest_["_child_nodes"][0][0][idx]
            else:
                idx = self.grf_forest_["_child_nodes"][0][1][idx]
        else:
            if x[varid] <= val:
                idx = self.grf_forest_["_child_nodes"][0][0][idx]
            else:
                idx = self.grf_forest_["_child_nodes"][0][1][idx]
        return [idx] + self._decision_path(x, idx)
