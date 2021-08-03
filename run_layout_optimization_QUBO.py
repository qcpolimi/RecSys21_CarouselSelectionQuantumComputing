#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 29/01/2021

@author: anonymous for blind review
"""

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import LazyFixedEmbeddingComposite

import pandas as pd
import dimod, os, traceback, neal, argparse
from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager import *
from Data_manager.DataSplitter_Holdout import DataSplitter_Holdout
from Data_manager.data_consistency_check import assert_disjoint_matrices
from Utils.RecommenderInstanceIterator import RecommenderInstanceIterator
from Recommenders.Recommender_import_list import *

from LayoutOptimizer.QUBOModelSelection import maximum_energy_delta, model_selection, QUBOModelSelection_Pearson, QUBOModelSelection_Spearman, QUBOModelSelection_Kendall, QUBOModelSelection_Accuracy, QUBOModelSelection_GiniDiff
from LayoutOptimizer.EvaluatorHoldoutSave import EvaluatorHoldoutSave
from LayoutOptimizer.GreedyLayout import CarouselLayoutOptimizer_IncrementalGreedy, CarouselLayoutOptimizer_Greedy
from LayoutOptimizer.exhaustive_search import compute_exact_search_multiprocessing
from LayoutOptimizer.evaluate_layout import evaluate_layout_parallel




def compute_global_recommendation_count(recommender_iterator, evaluator_validation):

    for index, (recommender_instance, recommender_name) in enumerate(recommender_iterator):

        print("Processing {}: {}".format(index, recommender_name))

        try:
            if not(os.path.isfile(recommendations_folder_path + "/" + recommender_name + ".zip")):
                recommender_instance.load_model(model_folder_path, file_name = recommender_name + "_best_model.zip")

                evaluator_validation.evaluateRecommender(recommender_instance, recommender_name)

        except:
            traceback.print_exc()




from Data_manager.DataPostprocessing_User_sample import DataPostprocessing_User_sample

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exact_search',     help="Compute optimal selection via exact search", type = bool, default = False)
    parser.add_argument('-q', '--use_qpu',          help="Use QPU instead of SA to solve the QUBO problem", type = bool, default = False)

    input_flags = parser.parse_args()
    print(input_flags)


    dataset_class = Movielens10MReader
    # dataset_class = NetflixPrizeReader
    dataset_reader = dataset_class()

    if dataset_class is NetflixPrizeReader:
        dataset_reader = DataPostprocessing_User_sample(dataset_reader, user_quota = 0.2)


    result_folder_path = "result_experiments/{}/".format(dataset_reader._get_dataset_name())
    data_folder_path = result_folder_path + "data/"
    model_folder_path = result_folder_path + "models/"
    recommendations_folder_path = result_folder_path + "recommendations/"

    dataSplitter = DataSplitter_Holdout(dataset_reader, user_wise = False, split_interaction_quota_list=[80, 10, 10])
    dataSplitter.load_data(save_folder_path=data_folder_path)

    URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()
    URM_train_last_test = URM_train + URM_validation


    # Ensure IMPLICIT data and disjoint test-train split
    assert_disjoint_matrices([URM_train, URM_validation, URM_test])

    # If directory does not exist, create
    if not os.path.exists(recommendations_folder_path):
        os.makedirs(recommendations_folder_path)

    metric_to_optimize = 'PRECISION'
    cutoff_list_validation = 10

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[cutoff_list_validation])
    evaluator_validation_save = EvaluatorHoldoutSave(URM_validation, cutoff_list=[cutoff_list_validation],
                                                output_folder_path=recommendations_folder_path)


    collaborative_algorithm_list = [
        Random,
        TopPop,
        GlobalEffects,
        UserKNNCFRecommender,
        ItemKNNCFRecommender,
        P3alphaRecommender,
        RP3betaRecommender,
        PureSVDRecommender,
        NMFRecommender,
        IALSRecommender,
        MatrixFactorization_BPR_Cython,
        MatrixFactorization_FunkSVD_Cython,
        # MatrixFactorization_AsySVD_Cython,
        EASE_R_Recommender,
        SLIM_BPR_Cython,
        SLIMElasticNetRecommender,
        UserKNNCBFRecommender,
        ItemKNNCBFRecommender,
        UserKNN_CFCBF_Hybrid_Recommender,
        ItemKNN_CFCBF_Hybrid_Recommender,
        ]

    KNN_similarity_to_report_list = ['cosine']#, 'dice', 'jaccard', 'asymmetric', 'tversky', 'euclidean']

    ICM_dict = dataSplitter.get_loaded_ICM_dict()

    if "ICM_year" in ICM_dict:
        del ICM_dict["ICM_year"]

    UCM_dict = dataSplitter.get_loaded_UCM_dict()

    # This object allows to iterate over instances of all recommender models: collaborative, content and hybrids
    recommender_iterator = RecommenderInstanceIterator(recommender_class_list = collaborative_algorithm_list,
                                                       KNN_similarity_list = KNN_similarity_to_report_list,
                                                       URM = URM_train,
                                                       ICM_dict = ICM_dict,
                                                       UCM_dict = UCM_dict,
                                                       )

    compute_global_recommendation_count(recommender_iterator, evaluator_validation_save)

    if input_flags.exact_search:
        for n_carousels in range(2,4):
            try:

                compute_exact_search_multiprocessing(recommender_iterator,
                                                     n_carousels,
                                                     result_folder_path,
                                                     model_folder_path,
                                                     URM_validation,
                                                     cutoff_list_validation)

            except (KeyboardInterrupt, SystemExit) as e:
                # If getting a interrupt, terminate without saving the exception
                raise e

            except:
                traceback.print_exc()







    ################################################################################################################
    ##################
    ##################          EXHAUSTIVE SEARCH AND GREEDY ALGORITHMS
    ##################
    ################################################################################################################

    num_best_to_evaluate = 1
    n_carousels_max = 10
    baseline_dataframe_name = "Comparison_Exhaustive_Greedy.csv"

    if not(os.path.isfile(result_folder_path + baseline_dataframe_name)):

        result_dataframe = pd.DataFrame()
        greedy_layout = CarouselLayoutOptimizer_Greedy(recommender_iterator, model_folder_path, evaluator_validation, metric_to_optimize)

        incremental_greedy_layout = CarouselLayoutOptimizer_IncrementalGreedy(recommender_iterator, model_folder_path, metric_to_optimize, URM_validation, cutoff_list_validation)
        _, result_recommender_name_list_greedy_layout = incremental_greedy_layout.get_layout(n_carousels_max)


        for n_carousels in range(2, n_carousels_max):

            print("Number of carousels: {}".format(n_carousels))

            try:

                exhaustive_search_dict = pd.read_csv(result_folder_path + "Exhaustive_search_{}_carousels.csv".format(n_carousels), index_col=0)
                best_samples_exhaustive = exhaustive_search_dict.nlargest(num_best_to_evaluate, columns=['PRECISION'])  # .squeeze()

                result_samples_exhaustive = evaluate_layout_parallel(best_samples_exhaustive, recommender_iterator, model_folder_path, URM_test, cutoff_list_validation)

                result_samples_exhaustive["n_carousels"] = n_carousels
                result_samples_exhaustive["model"] = "exhaustive"
                for index, _ in result_samples_exhaustive.iterrows():
                    result_samples_exhaustive.loc[index, "loss"] = best_samples_exhaustive.loc[index, 'PRECISION']

                result_dataframe = result_dataframe.append(result_samples_exhaustive, ignore_index=True, sort=True)
                result_dataframe.to_csv(result_folder_path + "Exhaustive_comparison.csv", index=True)

            except FileNotFoundError:
                print("Exact search results not available")



            _, result_recommender_name_list = greedy_layout.get_layout(n_carousels)
            result_recommender_name_list = {"variable_" + model_name:1 for model_name in result_recommender_name_list}

            best_sample_greedy = pd.DataFrame(result_recommender_name_list, index=[0])
            result_samples_greedy = evaluate_layout_parallel(best_sample_greedy, recommender_iterator, model_folder_path, URM_test, cutoff_list_validation)

            result_samples_greedy["n_carousels"] = n_carousels
            result_samples_greedy["model"] = "greedy"

            result_dataframe = result_dataframe.append(result_samples_greedy, ignore_index=True, sort=True)
            result_dataframe.sort_index(axis="columns", inplace=True)
            result_dataframe.to_csv(result_folder_path + baseline_dataframe_name, index=True)




            columns = {"variable_" + model_name:1 for model_name in result_recommender_name_list_greedy_layout[:n_carousels]}
            best_sample_greedy = pd.DataFrame(columns, index=[0])

            result_samples_greedy = evaluate_layout_parallel(best_sample_greedy, recommender_iterator, model_folder_path, URM_test, cutoff_list_validation)

            result_samples_greedy["n_carousels"] = n_carousels
            result_samples_greedy["model"] = "incremental_greedy"

            result_dataframe = result_dataframe.append(result_samples_greedy, ignore_index=True, sort=True)
            result_dataframe.sort_index(axis="columns", inplace=True)
            result_dataframe.to_csv(result_folder_path + baseline_dataframe_name, index=True)





    ################################################################################################################
    ##################
    ##################          QUBO MODEL SELECTION
    ##################
    ################################################################################################################


    result_dataframe = pd.DataFrame()
    timing_dataframe = pd.DataFrame()
    best_samples_qubo = pd.DataFrame()
    result_dataframe_name = "Comparison_QUBO"
    timing_dataframe_name = "Comparison_QUBO_timing"
    result_QUBO_path = result_folder_path + "QUBO_SA_alpha/"

    QUBO_model_dict = {
        # "pearson": QUBOModelSelection_Pearson(recommender_iterator=recommender_iterator,
        #                                   data_folder=recommendations_folder_path),
        # "spearman": QUBOModelSelection_Spearman(recommender_iterator=recommender_iterator,
        #                                   data_folder=recommendations_folder_path),
        # "kendall": QUBOModelSelection_Kendall(recommender_iterator=recommender_iterator,
        #                                   data_folder=recommendations_folder_path),
        "gini_diff": QUBOModelSelection_GiniDiff(recommender_iterator=recommender_iterator,
                                      data_folder=recommendations_folder_path),
        "accuracy":  QUBOModelSelection_Accuracy(recommender_iterator=recommender_iterator,
                                      recommendations_folder=recommendations_folder_path,
                                      model_folder=model_folder_path,
                                      evaluator_validation = evaluator_validation,
                                      cutoff = cutoff_list_validation),
    }


    # If directory does not exist, create
    if not os.path.exists(result_QUBO_path):
        os.makedirs(result_QUBO_path)

    for n_carousels in range(2, n_carousels_max):

        print("Number of carousels: {}".format(n_carousels))

        accuracy_component = {
            None: None,
            "accuracy":  QUBOModelSelection_Accuracy(recommender_iterator=recommender_iterator,
                                          recommendations_folder=recommendations_folder_path,
                                          model_folder=model_folder_path,
                                          evaluator_validation = evaluator_validation,
                                          cutoff = cutoff_list_validation),
        }


        for QUBO_label, QUBO_generator in QUBO_model_dict.items():

            BQM = QUBO_generator.get_BQM()
            variable_order = list(BQM.iter_variables())


            for alpha in [0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 0.9]:

                for label_accuracy, QUBO_accuracy in accuracy_component.items():

                    label = QUBO_label

                    if label_accuracy is not None:
                        label += "_" + label_accuracy
                        BQM_accuracy = QUBO_accuracy.get_BQM()

                        # Compute weighted sum of the two BQMs
                        BQM_diversity_np = BQM.to_numpy_matrix(variable_order=variable_order)
                        BQM_accuracy_np = BQM_accuracy.to_numpy_matrix(variable_order=variable_order)

                        BQM_np = BQM_accuracy_np*alpha + BQM_diversity_np*(1-alpha)

                        BQM = dimod.BinaryQuadraticModel.from_numpy_matrix(BQM_np, variable_order=variable_order)


                    # Add constraint to select the desired number of carousels

                    # Specify the penalty based on the maximum change in the objective
                    # that could occur by flipping a single variable.  This ensures
                    # that the ground state will satisfy the constraints.
                    penalty = maximum_energy_delta(BQM)

                    BQM.update(dimod.generators.combinations(variable_order,
                                                              n_carousels,
                                                              strength=penalty))


                    if input_flags.use_qpu:
                        # The LazyFixedEmbeddingComposite generates the problem embedding the first time and then re-uses it
                        # Since all QUBO matrices for this experiments are fully-connected, the embedding can be re-used
                        sampler = LazyFixedEmbeddingComposite(DWaveSampler())
                    else:
                        sampler = neal.SimulatedAnnealingSampler()


                    selection_dataframe, timing = model_selection(BQM = BQM,
                                                       recommender_iterator = recommender_iterator,
                                                       num_reads = 100,
                                                       n_carousels = n_carousels,
                                                       sampler = sampler,
                                                       )

                    selection_dataframe.to_csv(result_QUBO_path + "QUBO_selection_n_carousels_{}_{}_{}.csv".format(n_carousels, label, alpha), index=True)

                    timing["n_carousels"] = n_carousels
                    timing["model"] = label
                    timing["alpha"] = alpha
                    timing["k_penalty"] = penalty
                    timing_dataframe = timing_dataframe.append(timing, ignore_index=True, sort=True)
                    timing_dataframe.sort_index(axis="columns", inplace=True)
                    timing_dataframe.to_csv(result_QUBO_path + timing_dataframe_name + ".csv", index=True)


                    selection_dataframe = selection_dataframe[selection_dataframe["valid"]==1]
                    selection_dataframe["n_carousels"] = n_carousels
                    selection_dataframe["model"] = label
                    selection_dataframe["alpha"] = alpha
                    selection_dataframe["k_penalty"] = penalty
                    best_samples_qubo = best_samples_qubo.append(selection_dataframe.nsmallest(num_best_to_evaluate, columns=['energy']), ignore_index=True)




    result_dataframe_validation = evaluate_layout_parallel(best_samples_qubo, recommender_iterator, model_folder_path, URM_validation, cutoff_list_validation)
    result_dataframe_validation.sort_index(axis="columns", inplace=True)
    result_dataframe_validation.to_csv(result_QUBO_path + result_dataframe_name + "_validation.csv", index=True)

    print("Evaluation on Validation data complete")

    # Select the alpha value with best Precision on validation data
    idx = result_dataframe_validation.groupby(["model", "n_carousels"])["PRECISION"].transform(max) == result_dataframe_validation["PRECISION"]
    best_samples_qubo = best_samples_qubo[idx]

    # Evaluate on the test data
    result_dataframe_test = evaluate_layout_parallel(best_samples_qubo, recommender_iterator, model_folder_path, URM_test, cutoff_list_validation)
    result_dataframe_test.sort_index(axis="columns", inplace=True)
    result_dataframe_test.to_csv(result_QUBO_path + result_dataframe_name + ".csv", index=True)





    def print_summary_table(metric):

        baseline_dataframe = pd.read_csv(result_folder_path + baseline_dataframe_name, index_col=0)
        result_dataframe = pd.read_csv(result_QUBO_path + result_dataframe_name + ".csv", index_col=0)

        baseline_dataframe.sort_index(axis="columns", inplace=True)
        result_dataframe.sort_index(axis="columns", inplace=True)
        full_dataframe = baseline_dataframe.append(result_dataframe, ignore_index=True, sort=True)
        full_dataframe = full_dataframe[["n_carousels", "model", "alpha", metric]]

        pretty_table_dataframe = pd.DataFrame(columns=["model", *[n_carousels for n_carousels in range(2, n_carousels_max)]])

        for model_value, model_result in full_dataframe.groupby("model"):

            row_data = {
                "model": model_value,
            }

            for index in model_result.index:
                row_data[model_result.loc[index, "n_carousels"]] = model_result.loc[index, metric]

            pretty_table_dataframe = pretty_table_dataframe.append(row_data, ignore_index=True, sort=True)

        pretty_table_dataframe.to_csv(result_QUBO_path + "Summary_table_{}.csv".format(metric), index=True)



    print_summary_table("PRECISION")
    print_summary_table("COVERAGE_ITEM")

