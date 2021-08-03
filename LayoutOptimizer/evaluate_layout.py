#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/02/2021

@author: anonymous for blind review
"""


from EvaluatorMultipleCarousels import EvaluatorMultipleCarousels
import multiprocessing, os
from functools import partial


def _evaluate_single_layout(parallel_pool_args, variable_prefix, recommender_iterator, model_folder_path, URM_test, cutoff_list_validation):

    index, best_sample = parallel_pool_args

    # print("Evaluating best model {}/{}".format(count+1, len(best_samples)))

    model_name_list = [model_name[len(variable_prefix):] for model_name, value in best_sample.iteritems() if
                       model_name.startswith(variable_prefix) and value]

    instance_list = []

    print("Evaluating layout {}. Models selected: {}".format(index+1, model_name_list))

    for model_name in model_name_list:
        instance = recommender_iterator.get_instance_from_name(model_name)
        instance.load_model(model_folder_path, file_name=model_name + "_best_model.zip")

        instance_list.append(instance)

    evaluator_test = EvaluatorMultipleCarousels(URM_test,
                                              cutoff_list=[cutoff_list_validation],
                                              exclude_seen=True,
                                              carousel_recommender_list=instance_list[:-1])

    results_dict, results_run_string = evaluator_test.evaluateRecommender(instance_list[-1])

    return index, results_dict



def evaluate_layout_parallel(best_samples, recommender_iterator, model_folder_path, URM_test, cutoff_list_validation):

    variable_prefix = "variable_"
    # dataframe_model_columns = [model_name for model_name in best_samples.columns if model_name.startswith(variable_prefix)]
    # result_dataframe = best_samples[dataframe_model_columns].copy()
    result_dataframe = best_samples.copy()

    _evaluate_single_layout_partial = partial(_evaluate_single_layout,
                                                   variable_prefix = variable_prefix,
                                                   recommender_iterator = recommender_iterator,
                                                   model_folder_path = model_folder_path,
                                                   URM_test = URM_test,
                                                   cutoff_list_validation = cutoff_list_validation)


    pool = multiprocessing.Pool(processes=int(os.cpu_count()*0.9), maxtasksperchild=1)
    result_list = pool.map(_evaluate_single_layout_partial, best_samples.iterrows())

    pool.close()
    pool.join()


    for index, results_dict in result_list:
        for metric_name, metric_value in results_dict[cutoff_list_validation].items():
            result_dataframe.loc[index, metric_name] = metric_value


    return result_dataframe

