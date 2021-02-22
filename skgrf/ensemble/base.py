import numpy as np


class GRFValidationMixin:
    def _check_cluster(self, X, cluster):
        """Validate cluster definitions against training data."""
        if cluster is None:
            self.classes_ = None
            self.n_classes_ = None
            return np.array([])
        if len(cluster) != X.shape[0]:
            raise ValueError("cluster length must be the same as X")
        self.classes_, new_cluster = np.unique(cluster, return_inverse=True)
        self.n_classes_ = len(self.classes_)
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
        if self.mtry is None:
            return min(int(np.ceil(np.sqrt(X.shape[1] + 20))), X.shape[1])
        else:
            return self.mtry

    def _check_sample_fraction(self, oob=False):
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
        if self.alpha <= 0 or self.alpha >= 0.25:
            raise ValueError("alpha must be between 0 and 0.25")

    def _check_reduced_form_weight(self):
        if self.reduced_form_weight < 0 or self.reduced_form_weight > 1:
            raise ValueError("reduced_form_weight must be between 0 and 1")

    def _check_boost_error_reduction(self):
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
