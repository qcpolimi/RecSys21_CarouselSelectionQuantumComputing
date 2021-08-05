#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/02/2021

@author: anonymous for blind review
"""

from EvaluatorMultipleCarousels import EvaluatorMultipleCarousels
from scipy.special import binom
from Recommenders.DataIO import DataIO
import os, time, traceback, multiprocessing, itertools
from functools import partial
import pandas as pd


def _evaluate_single_combination(layout_list, recommender_name_to_index_dict, recommender_name_list, n_carousels,
                                 temp_folder_path, model_folder_path, URM_validation, cutoff_list_validation):

    try:
        dataIO = DataIO(folder_path=temp_folder_path)

        instance_list = [instance for instance, _ in layout_list]
        file_name_list = [name for _, name in layout_list]

        # Skip if it already exists combinations with duplicates
        file_name = "_".join(str(recommender_name_to_index_dict[recommender_name]) for recommender_name in file_name_list)
        if os.path.isfile(temp_folder_path + file_name + ".zip"):
            return

        # Ignore combinations with duplicates
        if (len(set(file_name_list)) < len(file_name_list)):
            return

        # total = int(binom(len(recommender_name_list) + n_carousels -1, n_carousels))
        total = int(binom(len(recommender_name_list), n_carousels))
        complete = len(os.listdir(temp_folder_path))-1

        print("Exact search, Complete: {}/{} ({:.2f}%)".format(complete, total, complete / total * 100))

        start_time = time.time()

        for index, instance in enumerate(instance_list):
            instance.load_model(model_folder_path, file_name=file_name_list[index] + "_best_model.zip")
            instance_list[index] = instance

        evaluator_validation = EvaluatorMultipleCarousels(URM_validation,
                                                          cutoff_list=[cutoff_list_validation],
                                                          exclude_seen=True,
                                                          carousel_recommender_list=instance_list[:-1])

        results_df, results_run_string = evaluator_validation.evaluateRecommender(instance_list[-1])

        dataframe_row = {
            **{"variable_{}".format(recommender_name): 0 for recommender_name in recommender_name_list},
            **{metric: value for metric, value in results_df.loc[cutoff_list_validation].iteritems()},
            "cutoff": cutoff_list_validation,
            "evaluation time": time.time() - start_time,
        }

        for index, recommender_name in enumerate(file_name_list):
            dataframe_row["variable_{}".format(recommender_name)] = index + 1

        dataIO.save_data(file_name, data_dict_to_save={"dataframe_row": dataframe_row})

    except (KeyboardInterrupt, SystemExit) as e:
        # If getting a interrupt, terminate without saving the exception
        raise e

    except:
        traceback.print_exc()




def compute_exact_search_multiprocessing(recommender_iterator,
                                         n_carousels,
                                         result_folder_path,
                                         model_folder_path,
                                         URM_validation,
                                         cutoff_list_validation):

    if os.path.isfile(result_folder_path + "Exhaustive_search_{}_carousels.csv".format(n_carousels)):
        return

    # dataIO = DataIO(folder_path=output_folder_path)
    temp_folder_path = result_folder_path + "temp_{}_carousels/".format(n_carousels)

    # If directory does not exist, create
    if not os.path.exists(temp_folder_path):
        os.makedirs(temp_folder_path)

    dataIO_temp = DataIO(folder_path=temp_folder_path)
    recommender_name_list = [recommender_name for _,recommender_name in recommender_iterator]

    try:
        recommender_index_map = dataIO_temp.load_data("recommender_index_to_name_dict")
        recommender_name_to_index_dict = recommender_index_map["recommender_name_to_index_dict"]
        assert len(recommender_name_list) == len(recommender_name_to_index_dict)
        print("Mapping data found")

    except FileNotFoundError:
        print("Mapping data NOT found")
        recommender_index_to_name_dict = {index:recommender_name for index,recommender_name in enumerate(recommender_name_list)}
        recommender_name_to_index_dict = {value:key for key, value in recommender_index_to_name_dict.items()}

        dataIO_temp.save_data("recommender_index_to_name_dict",
                              data_dict_to_save={"recommender_index_to_name_dict":recommender_index_to_name_dict,
                                                 "recommender_name_to_index_dict":recommender_name_to_index_dict})



    _evaluate_single_combination_partial = partial(_evaluate_single_combination,
                                                   recommender_name_to_index_dict = recommender_name_to_index_dict,
                                                   recommender_name_list = recommender_name_list,
                                                   n_carousels = n_carousels,
                                                   temp_folder_path = temp_folder_path,
                                                   model_folder_path = model_folder_path,
                                                   URM_validation = URM_validation,
                                                   cutoff_list_validation = cutoff_list_validation)

    cases = list(itertools.combinations(recommender_iterator, n_carousels))
    cases = cases[-int(len(cases)*0.8):]
    # cases.reverse()
    pool = multiprocessing.Pool(processes=int(os.cpu_count()*0.4), maxtasksperchild=1)
    pool.map(_evaluate_single_combination_partial, cases)

    pool.close()
    pool.join()

    #
    # for index, (layout_list) in enumerate(itertools.combinations(recommender_iterator, n_carousels)):
    #     _evaluate_single_combination_partial(layout_list)



    # df_columns = [*recommender_name_list, 'energy', 'num_occurrences']
    result_dataframe = pd.DataFrame()



    for index, (layout_list) in enumerate(itertools.combinations(recommender_iterator, n_carousels)):
        file_name_list = [name for _, name in layout_list]
        file_name = "_".join(str(recommender_name_to_index_dict[recommender_name]) for recommender_name in file_name_list)

        # Ignore combinations with duplicates
        if (len(set(file_name_list)) < len(file_name_list)):
            return

        dataframe_row = dataIO_temp.load_data(file_name + ".zip")

        result_dataframe = result_dataframe.append(dataframe_row["dataframe_row"], ignore_index=True)

    result_dataframe.to_csv(result_folder_path + "Exhaustive_search_{}_carousels.csv".format(n_carousels), index=True)

    return result_dataframe


