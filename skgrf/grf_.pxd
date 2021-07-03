from libcpp.memory cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.set cimport set
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector


cdef extern from "./grf/src/commons/Data.h" namespace "grf":
    cdef cppclass Data:
        Data(
            const double* data_ptr,
            size_t num_rows,
            size_t num_cols,
        ) except +
        void set_outcome_index(size_t index)
        void set_treatment_index(size_t index)
        void set_instrument_index(size_t index)
        void set_weight_index(size_t index)
        void set_causal_survival_numerator_index(size_t index)
        void set_causal_survival_denominator_index(size_t index)
        void set_censor_index(size_t index)
        vector[size_t] get_all_values(
            vector[double]& all_values,
            vector[size_t]& sorted_samples,
            const vector[size_t]& samples,
            size_t var
        ) const
        size_t get_num_cols() const
        size_t get_num_rows() const
        size_t get_num_outcomes() const
        size_t get_num_treatments() const
        const set[size_t]& get_disallowed_split_variables const
        double get_outcome(size_t row) const
        double get_treatment(size_t row) const
        double get_instrument(size_t row) const
        double get_weight(size_t row) const
        double get_causal_survival_numerator(size_t row) const
        double get_causal_survival_denominator(size_t row) const
        bool is_failure(size_t row) const
        double get(size_t row, size_t col) const


cdef extern from "./grf/src/commons/utility.h" namespace "grf":
    void split_sequence(
        vector[unsigned int]& result,
        unsigned int start,
        unsigned int end,
        unsigned int num_parts,
    )


cdef extern from "./grf/src/forest/Forest.h" namespace "grf":
    cdef cppclass Forest:
        Forest(
            vector[unique_ptr[Tree]]& trees,
            size_t num_variables,
            size_t ci_group_size,
        )
        Forest(Forest&& forest) except +
        const size_t get_ci_group_size() const
        const size_t get_num_variables() const
        const vector[unique_ptr[Tree]]& get_trees() const
        vector[unique_ptr[Tree]]& get_trees_()


cdef extern from "./grf/src/forest/ForestOptions.h" namespace "grf":
    cdef cppclass ForestOptions:
        ForestOptions() except +
        ForestOptions(
            unsigned int num_trees,
            size_t ci_group_size,
            double sample_fraction,
            unsigned int mtry,
            unsigned int min_node_size,
            bool honesty,
            double honesty_fraction,
            bool honesty_prune_leaves,
            double alpha,
            double imbalance_penalty,
            unsigned int num_threads,
            unsigned int random_seed,
            const vector[size_t]& sample_clusters,
            unsigned int samples_per_cluster,
        )


cdef extern from "./grf/src/prediction/collector/DefaultPredictionCollector.h" namespace "grf":
    cdef cppclass DefaultPredictionCollector:
        DefaultPredictionCollector() except +


cdef extern from "./grf/src/prediction/collector/OptimizedPredictionCollector.h" namespace "grf":
    cdef cppclass OptimizedPredictionCollector:
        OptimizedPredictionCollector() except +


cdef extern from "./grf/src/prediction/collector/SampleWeightComputer.h" namespace "grf":
    cdef cppclass SampleWeightComputer:
        SampleWeightComputer() except +
        unordered_map[size_t, double] compute_weights(
            size_t sample,
            const Forest& forest,
            const vector[vector[size_t]]& leaf_nodes_by_tree,
            const vector[vector[bool]]& valid_trees_by_sample,
        ) const


cdef extern from "./grf/src/prediction/collector/TreeTraverser.h" namespace "grf":
    cdef cppclass TreeTraverser:
        TreeTraverser(unsigned int num_threads) except +
        vector[vector[size_t]] get_leaf_nodes(
            const Forest& forest,
            const Data& data,
            bool oob_prediction
        ) const
        vector[vector[bool]] get_valid_trees_by_sample(
            const Forest& forest,
            const Data& data,
            bool oob_prediction
        ) const


cdef extern from "./grf/src/prediction/DefaultPredictionStrategy.h" namespace "grf":
    cdef cppclass DefaultPredictionStrategy:
        DefaultPredictionStrategy() except +


cdef extern from "./grf/src/prediction/CausalSurvivalPredictionStrategy.h" namespace "grf":
    cdef cppclass CausalSurvivalPredictionStrategy:
        CausalSurvivalPredictionStrategy() except +


cdef extern from "./grf/src/prediction/ProbabilityPredictionStrategy.h" namespace "grf":
    cdef cppclass ProbabilityPredictionStrategy:
        ProbabilityPredictionStrategy() except +


cdef extern from "./grf/src/prediction/MultiRegressionPredictionStrategy.h" namespace "grf":
    cdef cppclass MultiRegressionPredictionStrategy:
        MultiRegressionPredictionStrategy() except +


cdef extern from "./grf/src/prediction/InstrumentalPredictionStrategy.h" namespace "grf":
    cdef cppclass InstrumentalPredictionStrategy:
        InstrumentalPredictionStrategy() except +


cdef extern from "./grf/src/prediction/LLCausalPredictionStrategy.h" namespace "grf":
    cdef cppclass LLCausalPredictionStrategy:
        LLCausalPredictionStrategy() except +


cdef extern from "./grf/src/prediction/MultiCausalPredictionStrategy.h" namespace "grf":
    cdef cppclass MultiCausalPredictionStrategy:
        MultiCausalPredictionStrategy() except +


cdef extern from "./grf/src/prediction/LocalLinearPredictionStrategy.h" namespace "grf":
    cdef cppclass LocalLinearPredictionStrategy:
        LocalLinearPredictionStrategy() except +


cdef extern from "./grf/src/prediction/ObjectiveBayesDebiaser.h" namespace "grf":
    cdef cppclass ObjectiveBayesDebiaser:
        ObjectiveBayesDebiaser() except +


cdef extern from "./grf/src/prediction/Prediction.h" namespace "grf":
    cdef cppclass Prediction:
        Prediction(const vector[double]& predictions)
        Prediction(
            const vector[double]& predictions,
            const vector[double]& variance_estimates,
            const vector[double]& error_estimates,
            const vector[double]& excess_error_estimates,
        )
        const vector[double]& get_predictions() const
        const vector[double]& get_variance_estimates() const
        const vector[double]& get_error_estimates() const
        const vector[double]& get_excess_error_estimates() const
        const bool contains_variance_estimates() const
        const bool contains_error_estimates() const
        const size_t size() const


cdef extern from "./grf/src/prediction/PredictionValues.h" namespace "grf":
    cdef cppclass PredictionValues:
        PredictionValues() except +
        PredictionValues(
            const vector[vector[double]]& values,
            size_t num_types,
        )
        const vector[vector[double]]& get_all_values() const
        const size_t get_num_types() const


cdef extern from "./grf/src/prediction/OptimizedPredictionStrategy.h" namespace "grf":
    cdef cppclass OptimizedPredictionStrategy:
        OptimizedPredictionStrategy() except +


cdef extern from "./grf/src/prediction/QuantilePredictionStrategy.h" namespace "grf":
    cdef cppclass QuantilePredictionStrategy:
        QuantilePredictionStrategy() except +


cdef extern from "./grf/src/prediction/RegressionPredictionStrategy.h" namespace "grf":
    cdef cppclass RegressionPredictionStrategy:
        RegressionPredictionStrategy() except +


cdef extern from "./grf/src/prediction/SurvivalPredictionStrategy.h" namespace "grf":
    cdef cppclass SurvivalPredictionStrategy:
        SurvivalPredictionStrategy() except +


cdef extern from "./grf/src/relabeling/InstrumentalRelabelingStrategy.h" namespace "grf":
    cdef cppclass InstrumentalRelabelingStrategy:
        InstrumentalRelabelingStrategy() except +


cdef extern from "./grf/src/relabeling/CausalSurvivalRelabelingStrategy.h" namespace "grf":
    cdef cppclass CausalSurvivalRelabelingStrategy:
        CausalSurvivalRelabelingStrategy() except +


cdef extern from "./grf/src/relabeling/MultiCausalRelabelingStrategy.h" namespace "grf":
    cdef cppclass MultiCausalRelabelingStrategy:
        MultiCausalRelabelingStrategy() except +


cdef extern from "./grf/src/relabeling/LLRegressionRelabelingStrategy.h" namespace "grf":
    cdef cppclass LLRegressionRelabelingStrategy:
        LLRegressionRelabelingStrategy() except +


cdef extern from "./grf/src/relabeling/NoopRelabelingStrategy.h" namespace "grf":
    cdef cppclass NoopRelabelingStrategy:
        NoopRelabelingStrategy() except +


cdef extern from "./grf/src/relabeling/MultiNoopRelabelingStrategy.h" namespace "grf":
    cdef cppclass MultiNoopRelabelingStrategy:
        MultiNoopRelabelingStrategy() except +


cdef extern from "./grf/src/relabeling/QuantileRelabelingStrategy.h" namespace "grf":
    cdef cppclass QuantileRelabelingStrategy:
        QuantileRelabelingStrategy() except +


cdef extern from "./grf/src/relabeling/RelabelingStrategy.h" namespace "grf":
    cdef cppclass RelabelingStrategy:
        RelabelingStrategy() except +


cdef extern from "./grf/src/sampling/RandomSampler.h" namespace "grf":
    cdef cppclass RandomSampler:
        RandomSampler() except +


cdef extern from "./grf/src/sampling/SamplingOptions.h" namespace "grf":
    cdef cppclass SamplingOptions:
        SamplingOptions() except +


cdef extern from "./grf/src/splitting/factory/InstrumentalSplittingRuleFactory.h" namespace "grf":
    cdef cppclass InstrumentalSplittingRuleFactory:
        InstrumentalSplittingRuleFactory() except +


cdef extern from "./grf/src/splitting/factory/CausalSurvivalSplittingRuleFactory.h" namespace "grf":
    cdef cppclass CausalSurvivalSplittingRuleFactory:
        CausalSurvivalSplittingRuleFactory() except +


cdef extern from "./grf/src/splitting/factory/MultiCausalSplittingRuleFactory.h" namespace "grf":
    cdef cppclass MultiCausalSplittingRuleFactory:
        MultiCausalSplittingRuleFactory() except +


cdef extern from "./grf/src/splitting/factory/MultiRegressionSplittingRuleFactory.h" namespace "grf":
    cdef cppclass MultiRegressionSplittingRuleFactory:
        MultiRegressionSplittingRuleFactory() except +


cdef extern from "./grf/src/splitting/factory/RegressionSplittingRuleFactory.h" namespace "grf":
    cdef cppclass RegressionSplittingRuleFactory:
        RegressionSplittingRuleFactory() except +


cdef extern from "./grf/src/splitting/factory/ProbabilitySplittingRuleFactory.h" namespace "grf":
    cdef cppclass ProbabilitySplittingRuleFactory:
        ProbabilitySplittingRuleFactory() except +


cdef extern from "./grf/src/splitting/factory/SplittingRuleFactory.h" namespace "grf":
    cdef cppclass SplittingRuleFactory:
        SplittingRuleFactory() except +


cdef extern from "./grf/src/splitting/factory/SurvivalSplittingRuleFactory.h" namespace "grf":
    cdef cppclass SurvivalSplittingRuleFactory:
        SurvivalSplittingRuleFactory() except +


cdef extern from "./grf/src/splitting/InstrumentalSplittingRule.h" namespace "grf":
    cdef cppclass InstrumentalSplittingRule:
        InstrumentalSplittingRule() except +


cdef extern from "./grf/src/splitting/CausalSurvivalSplittingRule.h" namespace "grf":
    cdef cppclass CausalSurvivalSplittingRule:
        CausalSurvivalSplittingRule() except +


cdef extern from "./grf/src/splitting/MultiCausalSplittingRule.h" namespace "grf":
    cdef cppclass MultiCausalSplittingRule:
        MultiCausalSplittingRule() except +


cdef extern from "./grf/src/splitting/MultiRegressionSplittingRule.h" namespace "grf":
    cdef cppclass MultiRegressionSplittingRule:
        MultiRegressionSplittingRule() except +


cdef extern from "./grf/src/splitting/ProbabilitySplittingRule.h" namespace "grf":
    cdef cppclass ProbabilitySplittingRule:
        ProbabilitySplittingRule() except +


cdef extern from "./grf/src/splitting/RegressionSplittingRule.h" namespace "grf":
    cdef cppclass RegressionSplittingRule:
        RegressionSplittingRule() except +


cdef extern from "./grf/src/splitting/SurvivalSplittingRule.h" namespace "grf":
    cdef cppclass SurvivalSplittingRule:
        SurvivalSplittingRule() except +


cdef extern from "./grf/src/tree/Tree.h" namespace "grf":
    cdef cppclass Tree:
        Tree(
            size_t root_node,
            const vector[vector[size_t]]& child_nodes,
            const vector[vector[size_t]]& leaf_samples,
            const vector[size_t]& split_vars,
            const vector[double]& split_values,
            const vector[size_t]& drawn_samples,
            const vector[bool]& send_missing_left,
            const PredictionValues& prediction_values,
        )
        size_t get_root_node() const
        const vector[vector[size_t]]& get_child_nodes() const
        const vector[vector[size_t]]& get_leaf_samples() const
        const vector[size_t]& get_split_vars() const
        const vector[double]& get_split_values() const
        const vector[size_t]& get_drawn_samples() const
        const vector[bool]& get_send_missing_left() const
        const PredictionValues& get_prediction_values() const


cdef extern from "./grf/src/tree/TreeOptions.h" namespace "grf":
    cdef cppclass TreeOptions:
        TreeOptions() except +


cdef extern from "./grf/src/tree/TreeTrainer.h" namespace "grf":
    cdef cppclass TreeTrainer:
        TreeTrainer() except +


cdef extern from "./grf/src/forest/ForestPredictor.h" namespace "grf":
    cdef cppclass ForestPredictor:
        ForestPredictor(ForestPredictor&)
        ForestPredictor(
            unsigned int num_threads,
            unique_ptr[DefaultPredictionStrategy] strategy,
        )
        ForestPredictor(
            unsigned int num_threads,
            unique_ptr[OptimizedPredictionStrategy] strategy,
        )
        vector[Prediction] predict(
            const Forest& forest,
            const Data& train_data,
            const Data& data,
            bool estimate_variance
        ) const
        vector[Prediction] predict_oob(
            const Forest& forest,
            const Data& data,
            bool estimate_variance,
        ) const


cdef extern from "./grf/src/forest/ForestPredictors.h" namespace "grf":
    cdef ForestPredictor instrumental_predictor(unsigned int num_threads)
    cdef ForestPredictor ll_regression_predictor(
        unsigned int num_threads,
        vector[double] lambdas,
        bool weight_penalty,
        vector[size_t] linear_correction_variables,
    )
    cdef ForestPredictor probability_predictor(
        unsigned int num_threads,
        size_t num_classes,
    )
    cdef ForestPredictor quantile_predictor(
        unsigned int num_threads,
        const vector[double]& quantiles,
    )
    cdef ForestPredictor regression_predictor(unsigned int num_threads)
    cdef ForestPredictor survival_predictor(
        unsigned int num_threads,
        size_t num_failures,
        int prediction_type,
    )


cdef extern from "./grf/src/forest/ForestTrainer.h" namespace "grf":
    cdef cppclass ForestTrainer:
        # no nullary constructor
        # and we have to explicitly declare the copy constructor
        # so that we can instantiate with regression_trainer()
        ForestTrainer(ForestTrainer&)
        ForestTrainer(
            unique_ptr[RelabelingStrategy] relabeling_strategy,
            unique_ptr[SplittingRuleFactory] splitting_rule_factory,
            unique_ptr[OptimizedPredictionStrategy] prediction_strategy,
        )
        Forest train(
            const Data& data,
            const ForestOptions& options,
        )


cdef extern from "./grf/src/forest/ForestTrainers.h" namespace "grf":
    cdef ForestTrainer instrumental_trainer(
        double reduced_form_weight,
        bool stabilize_splits,
    )
    cdef ForestTrainer ll_regression_trainer(
        double split_lambda,
        bool weight_penalty,
        const vector[double]& overall_beta,
        size_t ll_split_cutoff,
        vector[size_t] ll_split_variables,
    )
    cdef ForestTrainer probability_trainer(size_t num_classes)
    cdef ForestTrainer quantile_trainer(const vector[double]& quantiles)
    cdef ForestTrainer regression_trainer()
    cdef ForestTrainer survival_trainer()


cdef extern from "./grf/src/analysis/SplitFrequencyComputer.h" namespace "grf":
    cdef cppclass SplitFrequencyComputer:
        vector[vector[size_t]] compute(
            const Forest& forest,
            size_t max_depth
        )