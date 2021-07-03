import cython
import numpy as np
import sys

cimport numpy as np
from cython.operator cimport dereference as deref
from cython.operator cimport postincrement as postinc
from libcpp.memory cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport move
from libcpp.vector cimport vector

from skgrf cimport grf_


cdef class Data:
    """Cython wrapper for Data C++ class.

    This wraps the Data class in C++, which encapsulates training data passed to the
    random forest classes. It allows us to pass numpy arrays as a grf-compatible
    Data object.
    """
    cdef unique_ptr[grf_.Data] c_data

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self,
        np.ndarray[double, ndim=2, mode="fortran"] data not None,
    ):
        cdef size_t num_rows = np.PyArray_DIMS(data)[0]
        cdef size_t num_cols = np.PyArray_DIMS(data)[1]
        self.c_data.reset(
            new grf_.Data(
                &data[0, 0],
                num_rows,
                num_cols,
            )
        )

    def get(self, size_t row, size_t col):
        return deref(self.c_data).get(row, col)


cdef class GRFForest:
    """Cython wrapper for forest pointer.

    Holds a reference to the forest so that we don't have to
    deserialize on each predict call.
    """
    cdef grf_.Forest* forest

    def __cinit__(self, dict forest_object):
        self.forest = deserialize_forest(forest_object)


cpdef regression_train(
    np.ndarray[double, ndim=2, mode="fortran"] train_matrix,
    size_t outcome_index,
    size_t sample_weight_index,
    bool use_sample_weights,
    unsigned int mtry,
    unsigned int num_trees,
    unsigned int min_node_size,
    double sample_fraction,
    bool honesty,
    double honesty_fraction,
    bool honesty_prune_leaves,
    size_t ci_group_size,
    double alpha,
    double imbalance_penalty,
    vector[size_t] clusters,
    unsigned int samples_per_cluster,
    bool compute_oob_predictions,
    unsigned int num_threads,
    unsigned int seed,
):
    cdef grf_.ForestOptions* options
    cdef vector[grf_.Prediction] predictions

    trainer = new grf_.ForestTrainer(grf_.regression_trainer())
    data = Data(train_matrix)
    deref(data.c_data).set_outcome_index(outcome_index)

    if use_sample_weights:
        deref(data.c_data).set_weight_index(sample_weight_index - 1)

    options = new grf_.ForestOptions(
        num_trees,
        ci_group_size,
        sample_fraction,
        mtry,
        min_node_size,
        honesty,
        honesty_fraction,
        honesty_prune_leaves,
        alpha,
        imbalance_penalty,
        num_threads,
        seed,
        clusters,
        samples_per_cluster,
    )

    forest = new grf_.Forest(trainer.train(deref(data.c_data), deref(options)))

    if compute_oob_predictions:
        predictor = new grf_.ForestPredictor(grf_.regression_predictor(num_threads))
        predictions = predictor.predict_oob(deref(forest), deref(data.c_data), False)

    return create_forest_object(forest, predictions)


cpdef regression_predict(
    GRFForest forest_wrapper,
    np.ndarray[double, ndim=2, mode="fortran"] train_matrix,
    size_t outcome_index,
    np.ndarray[double, ndim=2, mode="fortran"] test_matrix,
    unsigned int num_threads,
    bool estimate_variance,
):
    cdef vector[grf_.Prediction] predictions

    predictor = new grf_.ForestPredictor(grf_.regression_predictor(num_threads))
    train_data = Data(train_matrix)
    deref(train_data.c_data).set_outcome_index(outcome_index)

    test_data = Data(test_matrix)

    predictions = predictor.predict(
        deref(forest_wrapper.forest),
        deref(train_data.c_data),
        deref(test_data.c_data),
        estimate_variance,
    )

    return create_prediction_object(predictions)


cpdef quantile_train(
    vector[double] quantiles,
    bool regression_splitting,
    np.ndarray[double, ndim=2, mode="fortran"] train_matrix,
    size_t outcome_index,
    unsigned int mtry,
    unsigned int num_trees,
    int min_node_size,
    double sample_fraction,
    bool honesty,
    double honesty_fraction,
    bool honesty_prune_leaves,
    size_t ci_group_size,
    double alpha,
    double imbalance_penalty,
    vector[size_t] clusters,
    unsigned int samples_per_cluster,
    bool compute_oob_predictions,
    int num_threads,
    unsigned int seed,
):
    cdef grf_.ForestOptions* options
    cdef vector[grf_.Prediction] predictions

    if regression_splitting:
        trainer = new grf_.ForestTrainer(grf_.regression_trainer())
    else:
        trainer = new grf_.ForestTrainer(grf_.quantile_trainer(quantiles))

    data = Data(train_matrix)
    deref(data.c_data).set_outcome_index(outcome_index)

    options = new grf_.ForestOptions(
        num_trees,
        ci_group_size,
        sample_fraction,
        mtry,
        min_node_size,
        honesty,
        honesty_fraction,
        honesty_prune_leaves,
        alpha,
        imbalance_penalty,
        num_threads,
        seed,
        clusters,
        samples_per_cluster,
    )

    forest = new grf_.Forest(trainer.train(deref(data.c_data), deref(options)))

    if compute_oob_predictions:
        predictor = new grf_.ForestPredictor(grf_.quantile_predictor(num_threads, quantiles))

    return create_forest_object(forest, predictions)


cpdef quantile_predict(
    GRFForest forest_wrapper,
    vector[double] quantiles,
    np.ndarray[double, ndim=2, mode="fortran"] train_matrix,
    size_t outcome_index,
    np.ndarray[double, ndim=2, mode="fortran"] test_matrix,
    unsigned int num_threads,
):
    cdef vector[grf_.Prediction] predictions

    predictor = new grf_.ForestPredictor(grf_.quantile_predictor(num_threads, quantiles))
    train_data = Data(train_matrix)
    deref(train_data.c_data).set_outcome_index(outcome_index)

    test_data = Data(test_matrix)

    predictions = predictor.predict(
        deref(forest_wrapper.forest),
        deref(train_data.c_data),
        deref(test_data.c_data),
        False,  # estimate_variance
    )

    return create_prediction_object(predictions)


cpdef survival_train(
    np.ndarray[double, ndim=2, mode="fortran"] train_matrix,
    size_t outcome_index,
    size_t censor_index,
    size_t sample_weight_index,
    bool use_sample_weights,
    unsigned int mtry,
    unsigned int num_trees,
    unsigned int min_node_size,
    double sample_fraction,
    bool honesty,
    double honesty_fraction,
    bool honesty_prune_leaves,
    double alpha,
    size_t num_failures,
    vector[size_t] clusters,
    unsigned int samples_per_cluster,
    bool compute_oob_predictions,
    int num_threads,
    unsigned int seed,
):
    cdef grf_.ForestOptions* options
    cdef vector[grf_.Prediction] predictions

    trainer = new grf_.ForestTrainer(grf_.survival_trainer())

    data = Data(train_matrix)
    deref(data.c_data).set_outcome_index(outcome_index)
    deref(data.c_data).set_censor_index(censor_index)
    if use_sample_weights:
        deref(data.c_data).set_weight_index(sample_weight_index)

    options = new grf_.ForestOptions(
        num_trees,
        1,  # ci_group_size
        sample_fraction,
        mtry,
        min_node_size,
        honesty,
        honesty_fraction,
        honesty_prune_leaves,
        alpha,
        0,  # imbalance_penalty
        num_threads,
        seed,
        clusters,
        samples_per_cluster,
    )

    forest = new grf_.Forest(trainer.train(deref(data.c_data), deref(options)))

    if compute_oob_predictions:
        predictor = new grf_.ForestPredictor(grf_.survival_predictor(num_threads, num_failures, 0))

    return create_forest_object(forest, predictions)


cpdef survival_predict(
    GRFForest forest_wrapper,
    np.ndarray[double, ndim=2, mode="fortran"] train_matrix,
    size_t outcome_index,
    size_t censor_index,
    size_t sample_weight_index,
    bool use_sample_weights,
    np.ndarray[double, ndim=2, mode="fortran"] test_matrix,
    unsigned int num_threads,
    size_t num_failures,
):
    cdef vector[grf_.Prediction] predictions

    predictor = new grf_.ForestPredictor(grf_.survival_predictor(num_threads, num_failures, 0))
    train_data = Data(train_matrix)
    deref(train_data.c_data).set_outcome_index(outcome_index)
    deref(train_data.c_data).set_censor_index(censor_index)
    if use_sample_weights:
        deref(train_data.c_data).set_weight_index(sample_weight_index)

    test_data = Data(test_matrix)

    predictions = predictor.predict(
        deref(forest_wrapper.forest),
        deref(train_data.c_data),
        deref(test_data.c_data),
        False,  # estimate_variance
    )

    return create_prediction_object(predictions)


cpdef ll_regression_train(
    np.ndarray[double, ndim=2, mode="fortran"] train_matrix,
    size_t outcome_index,
    size_t sample_weight_index,
    double ll_split_lambda,
    bool ll_split_weight_penalty,
    vector[size_t] ll_split_variables,
    size_t ll_split_cutoff,
    vector[double] overall_beta,
    bool use_sample_weights,
    unsigned int mtry,
    unsigned int num_trees,
    unsigned int min_node_size,
    double sample_fraction,
    bool honesty,
    double honesty_fraction,
    bool honesty_prune_leaves,
    size_t ci_group_size,
    double alpha,
    double imbalance_penalty,
    vector[size_t] clusters,
    unsigned int samples_per_cluster,
    unsigned int num_threads,
    unsigned int seed,
):
    cdef grf_.ForestOptions* options
    cdef vector[grf_.Prediction] predictions

    trainer = new grf_.ForestTrainer(
        grf_.ll_regression_trainer(
            ll_split_lambda,
            ll_split_weight_penalty,
            overall_beta,
            ll_split_cutoff,
            ll_split_variables,
        )
    )

    data = Data(train_matrix)
    deref(data.c_data).set_outcome_index(outcome_index)
    if use_sample_weights:
        deref(data.c_data).set_weight_index(sample_weight_index)

    options = new grf_.ForestOptions(
        num_trees,
        ci_group_size,
        sample_fraction,
        mtry,
        min_node_size,
        honesty,
        honesty_fraction,
        honesty_prune_leaves,
        alpha,
        imbalance_penalty,
        num_threads,
        seed,
        clusters,
        samples_per_cluster,
    )

    forest = new grf_.Forest(trainer.train(deref(data.c_data), deref(options)))

    return create_forest_object(forest, predictions)


cpdef ll_regression_predict(
    GRFForest forest_wrapper,
    np.ndarray[double, ndim=2, mode="fortran"] train_matrix,
    size_t outcome_index,
    np.ndarray[double, ndim=2, mode="fortran"] test_matrix,
    vector[double] ll_lambda,
    bool ll_weight_penalty,
    vector[size_t] linear_correction_variables,
    unsigned int num_threads,
    bool estimate_variance,
):
    cdef vector[grf_.Prediction] predictions

    predictor = new grf_.ForestPredictor(
        grf_.ll_regression_predictor(
            num_threads,
            ll_lambda,
            ll_weight_penalty,
            linear_correction_variables,
        )
    )
    train_data = Data(train_matrix)
    deref(train_data.c_data).set_outcome_index(outcome_index)

    test_data = Data(test_matrix)

    predictions = predictor.predict(
        deref(forest_wrapper.forest),
        deref(train_data.c_data),
        deref(test_data.c_data),
        estimate_variance,
    )

    return create_prediction_object(predictions)


cpdef instrumental_train(
    np.ndarray[double, ndim=2, mode="fortran"] train_matrix,
    size_t outcome_index,
    size_t treatment_index,
    size_t instrument_index,
    size_t sample_weight_index,
    bool use_sample_weights,
    unsigned int mtry,
    unsigned int num_trees,
    unsigned int min_node_size,
    double sample_fraction,
    bool honesty,
    double honesty_fraction,
    bool honesty_prune_leaves,
    size_t ci_group_size,
    double reduced_form_weight,
    double alpha,
    double imbalance_penalty,
    bool stabilize_splits,
    vector[size_t] clusters,
    unsigned int samples_per_cluster,
    bool compute_oob_predictions,
    unsigned int num_threads,
    unsigned int seed,
):
    cdef grf_.ForestOptions* options
    cdef vector[grf_.Prediction] predictions

    trainer = new grf_.ForestTrainer(
        grf_.instrumental_trainer(
            reduced_form_weight, stabilize_splits,
        )
    )

    data = Data(train_matrix)
    deref(data.c_data).set_outcome_index(outcome_index)
    deref(data.c_data).set_treatment_index(treatment_index)
    deref(data.c_data).set_instrument_index(instrument_index)
    if use_sample_weights:
        deref(data.c_data).set_weight_index(sample_weight_index)

    options = new grf_.ForestOptions(
        num_trees,
        ci_group_size,
        sample_fraction,
        mtry,
        min_node_size,
        honesty,
        honesty_fraction,
        honesty_prune_leaves,
        alpha,
        imbalance_penalty,
        num_threads,
        seed,
        clusters,
        samples_per_cluster,
    )

    forest = new grf_.Forest(trainer.train(deref(data.c_data), deref(options)))

    if compute_oob_predictions:
        predictor = new grf_.ForestPredictor(grf_.instrumental_predictor(num_threads))

    return create_forest_object(forest, predictions)


cpdef instrumental_predict(
    GRFForest forest_wrapper,
    np.ndarray[double, ndim=2, mode="fortran"] train_matrix,
    size_t outcome_index,
    size_t treatment_index,
    size_t instrument_index,
    np.ndarray[double, ndim=2, mode="fortran"] test_matrix,
    unsigned int num_threads,
    bool estimate_variance,
):
    cdef vector[grf_.Prediction] predictions

    predictor = new grf_.ForestPredictor(
        grf_.instrumental_predictor(
            num_threads,
        )
    )
    train_data = Data(train_matrix)
    deref(train_data.c_data).set_outcome_index(outcome_index)
    deref(train_data.c_data).set_treatment_index(treatment_index)
    deref(train_data.c_data).set_instrument_index(instrument_index)

    test_data = Data(test_matrix)

    predictions = predictor.predict(
        deref(forest_wrapper.forest),
        deref(train_data.c_data),
        deref(test_data.c_data),
        estimate_variance,
    )

    return create_prediction_object(predictions)


cpdef causal_train(
    np.ndarray[double, ndim=2, mode="fortran"] train_matrix,
    size_t outcome_index,
    size_t treatment_index,
    size_t sample_weight_index,
    bool use_sample_weights,
    unsigned int mtry,
    unsigned int num_trees,
    unsigned int min_node_size,
    double sample_fraction,
    bool honesty,
    double honesty_fraction,
    bool honesty_prune_leaves,
    size_t ci_group_size,
    double reduced_form_weight,
    double alpha,
    double imbalance_penalty,
    bool stabilize_splits,
    vector[size_t] clusters,
    unsigned int samples_per_cluster,
    bool compute_oob_predictions,
    unsigned int num_threads,
    unsigned int seed,
):
    cdef grf_.ForestOptions* options
    cdef vector[grf_.Prediction] predictions

    trainer = new grf_.ForestTrainer(
        grf_.instrumental_trainer(
            reduced_form_weight, stabilize_splits,
        )
    )

    data = Data(train_matrix)
    deref(data.c_data).set_outcome_index(outcome_index)
    deref(data.c_data).set_treatment_index(treatment_index)
    deref(data.c_data).set_instrument_index(treatment_index)
    if use_sample_weights:
        deref(data.c_data).set_weight_index(sample_weight_index)

    options = new grf_.ForestOptions(
        num_trees,
        ci_group_size,
        sample_fraction,
        mtry,
        min_node_size,
        honesty,
        honesty_fraction,
        honesty_prune_leaves,
        alpha,
        imbalance_penalty,
        num_threads,
        seed,
        clusters,
        samples_per_cluster,
    )

    forest = new grf_.Forest(trainer.train(deref(data.c_data), deref(options)))

    if compute_oob_predictions:
        predictor = new grf_.ForestPredictor(grf_.instrumental_predictor(num_threads))

    return create_forest_object(forest, predictions)


cpdef causal_predict(
    GRFForest forest_wrapper,
    np.ndarray[double, ndim=2, mode="fortran"] train_matrix,
    size_t outcome_index,
    size_t treatment_index,
    np.ndarray[double, ndim=2, mode="fortran"] test_matrix,
    unsigned int num_threads,
    bool estimate_variance,
):
    cdef vector[grf_.Prediction] predictions

    predictor = new grf_.ForestPredictor(
        grf_.instrumental_predictor(
            num_threads,
        )
    )
    train_data = Data(train_matrix)
    deref(train_data.c_data).set_outcome_index(outcome_index)
    deref(train_data.c_data).set_treatment_index(treatment_index)
    deref(train_data.c_data).set_instrument_index(treatment_index)

    test_data = Data(test_matrix)

    predictions = predictor.predict(
        deref(forest_wrapper.forest),
        deref(train_data.c_data),
        deref(test_data.c_data),
        estimate_variance,
    )

    return create_prediction_object(predictions)


cpdef probability_train(
    np.ndarray[double, ndim=2, mode="fortran"] train_matrix,
    size_t outcome_index,
    size_t sample_weight_index,
    bool use_sample_weights,
    size_t num_classes,
    unsigned int mtry,
    unsigned int num_trees,
    unsigned int min_node_size,
    double sample_fraction,
    bool honesty,
    double honesty_fraction,
    bool honesty_prune_leaves,
    size_t ci_group_size,
    double alpha,
    double imbalance_penalty,
    vector[size_t] clusters,
    unsigned int samples_per_cluster,
    bool compute_oob_predictions,
    unsigned int num_threads,
    unsigned int seed,
):
    cdef grf_.ForestOptions* options
    cdef vector[grf_.Prediction] predictions

    trainer = new grf_.ForestTrainer(grf_.probability_trainer(num_classes))
    data = Data(train_matrix)
    deref(data.c_data).set_outcome_index(outcome_index)

    if use_sample_weights:
        deref(data.c_data).set_weight_index(sample_weight_index - 1)

    options = new grf_.ForestOptions(
        num_trees,
        ci_group_size,
        sample_fraction,
        mtry,
        min_node_size,
        honesty,
        honesty_fraction,
        honesty_prune_leaves,
        alpha,
        imbalance_penalty,
        num_threads,
        seed,
        clusters,
        samples_per_cluster,
    )

    forest = new grf_.Forest(trainer.train(deref(data.c_data), deref(options)))

    if compute_oob_predictions:
        predictor = new grf_.ForestPredictor(grf_.probability_predictor(num_threads, num_classes))
        predictions = predictor.predict_oob(deref(forest), deref(data.c_data), False)

    return create_forest_object(forest, predictions)


cpdef probability_predict(
    GRFForest forest_wrapper,
    np.ndarray[double, ndim=2, mode="fortran"] train_matrix,
    size_t outcome_index,
    size_t num_classes,
    np.ndarray[double, ndim=2, mode="fortran"] test_matrix,
    unsigned int num_threads,
    bool estimate_variance,
):
    cdef vector[grf_.Prediction] predictions

    predictor = new grf_.ForestPredictor(grf_.probability_predictor(num_threads, num_classes))
    train_data = Data(train_matrix)
    deref(train_data.c_data).set_outcome_index(outcome_index)

    test_data = Data(test_matrix)

    predictions = predictor.predict(
        deref(forest_wrapper.forest),
        deref(train_data.c_data),
        deref(test_data.c_data),
        estimate_variance,
    )

    return create_prediction_object(predictions)



cdef create_forest_object(
    grf_.Forest* forest,
    vector[grf_.Prediction] predictions
):
    result = serialize_forest(forest)
    if not predictions.empty():
        result.update(create_prediction_object(predictions))
    return result


cdef grf_.Forest* deserialize_forest(
    dict forest_object
):
    cdef grf_.Forest* forest
    cdef vector[unique_ptr[grf_.Tree]] trees
    cdef grf_.Tree* tree
    cdef unique_ptr[grf_.Tree] treeptr

    cdef size_t num_trees = forest_object["num_trees"]

    trees.reserve(num_trees)

    for t in range(num_trees):
        tree = new grf_.Tree(
            forest_object["root_nodes"][t],
            forest_object["child_nodes"][t],
            forest_object["leaf_samples"][t],
            forest_object["split_vars"][t],
            forest_object["split_values"][t],
            forest_object["drawn_samples"][t],
            forest_object["send_missing_left"][t],
            grf_.PredictionValues(
                forest_object["pv_values"][t],
                forest_object["pv_num_types"][t]
            ),
        )
        treeptr.reset(tree)
        trees.push_back(
            move(treeptr)
        )
    forest = new grf_.Forest(
        trees,
        forest_object["num_variables"],
        forest_object["ci_group_size"]
    )
    return forest


cdef serialize_forest(
    grf_.Forest* forest,
):
    result = dict()
    result["ci_group_size"] = forest.get_ci_group_size()
    result["num_variables"] = forest.get_num_variables()
    num_trees = forest.get_trees().size()
    result["num_trees"] = num_trees

    result["root_nodes"] = []
    result["child_nodes"] = []
    result["leaf_samples"] = []
    result["split_vars"] = []
    result["split_values"] = []
    result["drawn_samples"] = []
    result["send_missing_left"] = []
    result["pv_values"] = []
    result["pv_num_types"] = []

    for k in range(num_trees):
        # this is awkward but it avoids a compilation issue on linux
        # related to overloading of get_trees_ :
        # error: call of overloaded ‘move(__gnu_cxx::__alloc_traits<std::allocator<std::unique_ptr<grf::Tree> > >::value_type&)’ is ambiguous
        #
        # assigning the tree doesn't work because Tree lacks a nullary constructor
        result["root_nodes"].append(deref(forest.get_trees()[k]).get_root_node())
        result["child_nodes"].append(deref(forest.get_trees()[k]).get_child_nodes())
        result["leaf_samples"].append(deref(forest.get_trees()[k]).get_leaf_samples())
        result["split_vars"].append(deref(forest.get_trees()[k]).get_split_vars())
        result["split_values"].append(deref(forest.get_trees()[k]).get_split_values())
        result["drawn_samples"].append(deref(forest.get_trees()[k]).get_drawn_samples())
        result["send_missing_left"].append(deref(forest.get_trees()[k]).get_send_missing_left())

        result["pv_values"].append(deref(forest.get_trees()[k]).get_prediction_values().get_all_values())
        result["pv_num_types"].append(deref(forest.get_trees()[k]).get_prediction_values().get_num_types())
    return result


cdef create_prediction_object(
    const vector[grf_.Prediction]& predictions,
):
    output = dict()
    output["predictions"] = create_prediction_matrix(predictions)
    output["variance_estimates"] = create_variance_matrix(predictions)
    output["debiased_error"] = create_error_matrix(predictions)
    output["excess_error"] = create_excess_error_matrix(predictions)
    return output


cdef create_prediction_matrix(
    const vector[grf_.Prediction]& predictions,
):
    cdef size_t prediction_length

    if predictions.empty():
        return []

    result = []
    for j in range(predictions.size()):
        values = []
        prediction_values = predictions[j].get_predictions()
        for k in range(prediction_values.size()):
            values.append(prediction_values[k])
        result.append(values)

    return result


cdef create_variance_matrix(
    const vector[grf_.Prediction]& predictions,
):
    cdef size_t prediction_length

    if predictions.empty():
        return []

    if not predictions.at(0).contains_variance_estimates():
        return []

    prediction_length = predictions.at(0).size()

    result = []
    for j in range(predictions.size()):
        values = []
        variance_estimate = predictions[j].get_variance_estimates()
        for k in range(variance_estimate.size()):
            values.append(variance_estimate[k])
        result.append(values)

    return result


cdef create_error_matrix(
    const vector[grf_.Prediction]& predictions,
):
    cdef size_t prediction_length

    if predictions.empty():
        return []

    if not predictions.at(0).contains_error_estimates():
        return []

    prediction_length = predictions.at(0).size()

    result = []
    for j in range(predictions.size()):
        values = []
        error_estimate = predictions[j].get_error_estimates()
        for k in range(error_estimate.size()):
            values.append(error_estimate[k])
        result.append(values)

    return result

cdef create_excess_error_matrix(
    const vector[grf_.Prediction]& predictions,
):
    cdef size_t prediction_length

    if predictions.empty():
        return []

    if not predictions.at(0).contains_error_estimates():
        return []

    prediction_length = predictions.at(0).size()

    result = []
    for j in range(predictions.size()):
        values = []
        error_estimate = predictions[j].get_excess_error_estimates()
        for k in range(error_estimate.size()):
            values.append(error_estimate[k])
        result.append(values)

    return result


cpdef compute_split_frequencies(
    GRFForest forest_wrapper,
    size_t max_depth,
):
    cdef grf_.SplitFrequencyComputer computer
    return computer.compute(deref(forest_wrapper.forest), max_depth)


cpdef compute_kernel_weights(
    GRFForest forest_wrapper,
    np.ndarray[double, ndim=2, mode="fortran"] test_matrix,
    int num_threads,
    bool oob_prediction,
):
    cdef grf_.SampleWeightComputer computer
    cdef unordered_map[size_t, double] calculated_weights
    cdef unordered_map[size_t, double].iterator it

    test_data = Data(test_matrix)

    tree_traverser = new grf_.TreeTraverser(num_threads)
    leaf_nodes_by_tree = tree_traverser.get_leaf_nodes(deref(forest_wrapper.forest), deref(test_data.c_data), oob_prediction)
    trees_by_sample = tree_traverser.get_valid_trees_by_sample(deref(forest_wrapper.forest), deref(test_data.c_data), oob_prediction)

    samples = []
    neighbors = []
    weights = []
    for sample in range(test_matrix.shape[0]):
        calculated_weights = computer.compute_weights(sample, deref(forest_wrapper.forest), leaf_nodes_by_tree, trees_by_sample)
        it = calculated_weights.begin()
        while (it != calculated_weights.end()):
            samples.append(sample)
            neighbors.append(deref(it).first)
            weights.append(deref(it).second)
            postinc(it)

    return samples, neighbors, weights


