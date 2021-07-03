import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from skgrf import grf


class GRFMixin:
    def __getstate__(self):
        """Serialize.

        The GRFForest pointer ref is not serializable, so
        we must pop it off of state here.
        """
        self.__dict__.pop("grf_forest_cpp_", None)
        return self.__dict__

    def __setstate__(self, state):
        """Deserialize.

        Unpickle and ensure that we deserialize the forest
        to a C++ pointer, so that predictions are fast.
        """
        self.__dict__ = state
        try:
            check_is_fitted(self)
            self._ensure_ptr()
        except NotFittedError:
            pass

    def _ensure_ptr(self):
        """Ensure that a pointer to the C++ forest exists."""
        if hasattr(self, "grf_forest_") and not hasattr(self, "grf_forest_cpp_"):
            self.grf_forest_cpp_ = grf.GRFForest(self.grf_forest_)

    def _check_cluster(self, X, cluster):
        """Validate cluster definitions against training data."""
        if cluster is None:
            self.clusters_ = None
            self.n_clusters_ = 1
            return np.array([])
        if len(cluster) != X.shape[0]:
            raise ValueError("cluster length must be the same as X")
        self.clusters_, new_cluster = np.unique(cluster, return_inverse=True)
        self.n_clusters_ = len(self.clusters_)
        return new_cluster

    def _check_equalize_cluster_weights(self, cluster, sample_weight):
        """Validate cluster weight against equalize cluster weight param."""
        if len(cluster) == 0:
            return 0
        _, counts = np.unique(cluster, return_counts=True)
        if self.equalize_cluster_weights:
            if sample_weight is not None:
                raise ValueError(
                    "Cannot use sample_weight when equalize_cluster_weights is True"
                )
            return min(counts)
        else:
            return max(counts)

    def _check_mtry(self, X):
        """Validate mtry."""
        if self.mtry is None:
            return min(int(np.ceil(np.sqrt(X.shape[1] + 20))), X.shape[1])
        else:
            return self.mtry

    def _check_sample_fraction(self, oob=False):
        """Validate sample fraction"""
        if hasattr(self, "ci_group_size") and self.ci_group_size >= 2:
            if self.sample_fraction <= 0 or self.sample_fraction > 0.5:
                raise ValueError(
                    "sample fraction must be between 0 and 0.5 when ci_group_size >= 2"
                )
        if self.sample_fraction <= 0 or self.sample_fraction > 1:
            raise ValueError(
                "sample fraction must be between 0 and 1 when ci_group_size == 1"
            )
        if oob and self.sample_fraction >= 1:
            raise ValueError(
                "sample fraction must be strictly less than 1 for oob predictions"
            )

    def _check_alpha(self):
        """Validate alpha."""
        if self.alpha <= 0 or self.alpha >= 0.25:
            raise ValueError("alpha must be between 0 and 0.25")

    def _check_reduced_form_weight(self):
        """Validate reduced form weight."""
        if self.reduced_form_weight < 0 or self.reduced_form_weight > 1:
            raise ValueError("reduced_form_weight must be between 0 and 1")

    def _check_boost_error_reduction(self):
        """Validate boost error reduction."""
        if self.boost_error_reduction < 0 or self.boost_error_reduction > 1:
            raise ValueError("reduced_form_weight must be between 0 and 1")

    def _get_num_threads(self):
        """Get GRF-expected num_threads value."""
        return max([self.n_jobs, 0])  # sklearn convention is -1 for all cpus, grf is 0

    def _create_train_matrices(
        self,
        X,
        y=None,
        sample_weight=None,
        treatment=None,
        instrument=None,
        censor=None,
    ):
        """Create a concatenated training matrix.

        GRF expects training data to be combined into a single matrix with reference variables
        that point to column indices. This creates that concatenated matrix and sets
        reference indices to be passed to GRF.
        """
        n_cols = X.shape[1]
        concats = [X]
        if y is not None:
            self.outcome_index_ = n_cols
            n_cols += 1
            concats.append(np.atleast_2d(y).T)
        if treatment is not None:
            self.treatment_index_ = n_cols
            n_cols += 1
            concats.append(np.atleast_2d(treatment).T)
        if instrument is not None:
            self.instrument_index_ = n_cols
            n_cols += 1
            concats.append(np.atleast_2d(instrument).T)
        if censor is not None:
            self.censor_index_ = n_cols
            n_cols += 1
            concats.append(np.atleast_2d(censor).T)
        self.sample_weight_index_ = 0
        if sample_weight is not None:
            self.sample_weight_index_ = n_cols
            n_cols += 1
            concats.append(np.atleast_2d(sample_weight).T)
        return np.concatenate(concats, axis=1)

    def _check_num_samples(self, X):
        """Validate sample fraction against dataset."""
        if len(X) * self.sample_fraction < 1:
            raise ValueError(
                "The sample fraction is too small, resulting in less than 1 sample for fitting."
            )

    def _set_sample_weights(self, sample_weight):
        """Set leaf weights for access in ``Tree``."""
        self.grf_forest_["leaf_weights"] = []
        for tree in self.grf_forest_["leaf_samples"]:
            self.grf_forest_["leaf_weights"].append([])
            for node in tree:
                self.grf_forest_["leaf_weights"][-1].append(
                    sum([sample_weight[idx] for idx in node])
                )

    def _get_sample_values(self, values):
        """Map the leaf samples to corresponding values.

        Used to get corresponding sample weights or targets.
        """
        mapped_values = []
        for tree in self.grf_forest_["leaf_samples"]:
            mapped_values.append([])
            for node in tree:
                mapped_values[-1].append([values[idx] for idx in node])
        return mapped_values

    def _get_values(self, left, right, idx, values):
        """Recursively sum the child values through a tree."""
        left_values = (
            values[idx]
            if left[idx] == 0
            else self._get_values(left, right, left[idx], values)
        )
        right_values = (
            values[idx]
            if right[idx] == 0
            else self._get_values(left, right, right[idx], values)
        )
        values[idx] = left_values + right_values
        return values[idx]

    def _set_node_values(self, y, sample_weight):
        """Set the node values for ``Tree.value``."""
        forest_values = self._get_sample_values(y)
        forest_weights = self._get_sample_values(sample_weight)
        self.grf_forest_["node_values"] = []
        for idx in range(self.grf_forest_["num_trees"]):
            left = self.grf_forest_["child_nodes"][idx][0]
            right = self.grf_forest_["child_nodes"][idx][1]
            root = self.grf_forest_["root_nodes"][idx]
            values = forest_values[idx]
            self._get_values(
                left,
                right,
                root,
                values,
            )
            weights = forest_weights[idx]
            self._get_values(
                left,
                right,
                root,
                weights,
            )
            values = [
                np.average(v, weights=w, axis=0) if v else np.nan
                for v, w in zip(values, weights)
            ]
            self.grf_forest_["node_values"].append(values)

    def _set_n_classes(self):
        """Set num classes for ``Tree.n_classes``."""
        # for accessing in Tree
        self.grf_forest_["n_classes"] = getattr(self, "n_classes_", 1)
