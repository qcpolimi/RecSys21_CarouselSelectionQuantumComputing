#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/02/2021

@author: anonymous for blind review
"""

from Recommenders.DataIO import DataIO
from Evaluation.Evaluator import _create_empty_metrics_dict, EvaluatorHoldout
from itertools import chain
import scipy.sparse as sps
import numpy as np

class EvaluatorHoldoutSave(EvaluatorHoldout):
    """EvaluatorHoldout"""

    EVALUATOR_NAME = "EvaluatorHoldoutSave"

    def __init__(self, URM_test_list, cutoff_list, min_ratings_per_user=1, exclude_seen=True,
                 diversity_object = None,
                 ignore_items = None,
                 ignore_users = None,
                 verbose=True,
                 output_folder_path = None):


        super(EvaluatorHoldoutSave, self).__init__(URM_test_list, cutoff_list,
                                               diversity_object = diversity_object,
                                               min_ratings_per_user =min_ratings_per_user, exclude_seen=exclude_seen,
                                               ignore_items = ignore_items, ignore_users = ignore_users,
                                               verbose = verbose)

        self._output_folder_path = output_folder_path
        self._dataIO = DataIO(folder_path = self._output_folder_path)
        self._file_name = None

    def evaluateRecommender(self, recommender_object, file_name):
        self._file_name = file_name
        super(EvaluatorHoldoutSave, self).evaluateRecommender(recommender_object)
        self._file_name = None


    def _run_evaluation_on_selected_users(self, recommender_object, users_to_evaluate, block_size = None):

        if block_size is None:
            block_size = min(1000, int(1e8/self.n_items))
            block_size = min(block_size, len(users_to_evaluate))


        results_dict = _create_empty_metrics_dict(self.cutoff_list,
                                                  self.n_items, self.n_users,
                                                  recommender_object.get_URM_train(),
                                                  self.URM_test,
                                                  self.ignore_items_ID,
                                                  self.ignore_users_ID,
                                                  self.diversity_object)


        if self.ignore_items_flag:
            recommender_object.set_items_to_ignore(self.ignore_items_ID)

        # Start from -block_size to ensure it to be 0 at the first block
        user_batch_start = 0
        user_batch_end = 0

        recommended_items_all_list = []

        while user_batch_start < len(users_to_evaluate):

            user_batch_end = user_batch_start + block_size
            user_batch_end = min(user_batch_end, len(users_to_evaluate))

            test_user_batch_array = np.array(users_to_evaluate[user_batch_start:user_batch_end])
            user_batch_start = user_batch_end

            # Compute predictions for a batch of users using vectorization, much more efficient than computing it one at a time
            recommended_items_batch_list, scores_batch = recommender_object.recommend(test_user_batch_array,
                                                                      remove_seen_flag=self.exclude_seen,
                                                                      cutoff = self.max_cutoff,
                                                                      remove_top_pop_flag=False,
                                                                      remove_custom_items_flag=self.ignore_items_flag,
                                                                      return_scores = True
                                                                     )

            recommended_items_all_list.extend(recommended_items_batch_list)

            results_dict = self._compute_metrics_on_recommendation_list(test_user_batch_array = test_user_batch_array,
                                                         recommended_items_batch_list = recommended_items_batch_list,
                                                         scores_batch = scores_batch,
                                                         results_dict = results_dict)

        user_index_list = [[index] * len(recommendation_list) for index, recommendation_list in enumerate(recommended_items_all_list)]
        row_list = list(chain.from_iterable(user_index_list))
        col_list = list(chain.from_iterable(recommended_items_all_list))

        URM_recommendations = sps.csr_matrix(([1]*len(row_list), (row_list, col_list)))

        assert URM_recommendations.nnz == len(row_list)

        data_to_save = {
            "URM_recommendations": URM_recommendations,
            "item_recommendation_count": results_dict[self.cutoff_list[0]]['COVERAGE_ITEM'].recommended_counter,
            "item_correct_recommendation_count": results_dict[self.cutoff_list[0]]['COVERAGE_ITEM_CORRECT'].recommended_counter
        }

        self._dataIO.save_data(self._file_name, data_dict_to_save=data_to_save)

        return results_dict

