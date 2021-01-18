import numpy as np


class GRFValidationMixin:
    def _check_cluster(self, X, cluster):
        """Validate cluster definitions against training data."""
        if cluster is None:
            return np.array([])
        # TODO assert integers
        # TODO convert to factors (0 to n values only)
        return np.array(cluster)

    def _check_equalize_cluster_weights(self, cluster, sample_weight):
        """Validate cluster weight against equalize cluster weight param."""
        if len(cluster) == 0:
            return 0
        _, counts = np.unique(cluster, return_counts=True)
        if self.equalize_cluster_weights:
            if sample_weight is None:
                raise ValueError("Cannot use sample_weight when equalize_cluster_weights is True")
            return min(counts)
        else:
            return max(counts)

    def _get_num_threads(self):
        """Get GRF-expected num_threads value."""
        return max([self.n_jobs, 0])  # sklearn convention is -1 for all cpus, grf is 0

    def _create_train_matrices(self, X, y=None, sample_weight=None, treatment=None, instrument=None, censor=None):
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
            concats.append(treatment)
        if instrument is not None:
            self.instrument_index_ = n_cols
            n_cols += 1
            concats.append(instrument)
        if censor is not None:
            self.censor_index_ = n_cols
            n_cols += 1
            concats.append(censor)
        self.sample_weight_index_ = 0
        if sample_weight is not None:
            self.sample_weight_index_ = n_cols
            concats.append(sample_weight)
        return np.concatenate(concats, axis=1)
