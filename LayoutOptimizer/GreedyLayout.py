#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/02/2021

@author: anonymous for blind review
"""

import numpy as np
from EvaluatorMultipleCarousels import EvaluatorMultipleCarousels

class CarouselLayoutOptimizer_Greedy():
    """CarouselLayoutOptimizer_Greedy"""

    def __init__(self, recommender_iterator, model_folder, evaluator_validation, metric_to_optimize):
        self._recommender_iterator = recommender_iterator
        self._model_folder = model_folder
        self._evaluator_validation = evaluator_validation
        self._metric_to_optimize = metric_to_optimize

        self._evaluate_models()

    def _evaluate_models(self):
        self._recommender_result = np.zeros(self._recommender_iterator.len())
        self._recommender_instance_list = [None] * self._recommender_iterator.len()
        self._recommender_name_list = [None] * self._recommender_iterator.len()

        for index, (recommender_instance, recommender_name) in enumerate(self._recommender_iterator):
            print("CarouselLayoutOptimizer_Greedy: Model {}/{}".format(index + 1, self._recommender_iterator.len()))

            recommender_instance.load_model(self._model_folder, file_name=recommender_name + "_best_model.zip")
            self._recommender_instance_list[index] = recommender_instance
            self._recommender_name_list[index] = recommender_name
            result_dict, _ = self._evaluator_validation.evaluateRecommender(recommender_instance)

            cutoff = list(result_dict.keys())[0]
            self._recommender_result[index] = result_dict[cutoff][self._metric_to_optimize]

    def get_layout(self, n_carousels):
        # Sort models
        recommender_argsort = np.argsort(-self._recommender_result)

        result_carousel_list = [self._recommender_instance_list[index] for index in recommender_argsort[0:n_carousels]]
        result_recommender_name_list = [self._recommender_name_list[index] for index in recommender_argsort[0:n_carousels]]

        return result_carousel_list, result_recommender_name_list







class CarouselLayoutOptimizer_IncrementalGreedy():
    """CarouselLayoutOptimizer_Greedy"""

    def __init__(self, recommender_iterator, model_folder, metric_to_optimize, URM_validation, cutoff):
        self._recommender_iterator = recommender_iterator
        self._model_folder = model_folder
        self._metric_to_optimize = metric_to_optimize
        self._URM_validation = URM_validation
        self._cutoff = cutoff

    def _evaluate_models(self, carousel_list):
        self._recommender_result = np.zeros(self._recommender_iterator.len())
        self._recommender_instance_list = [None] * self._recommender_iterator.len()
        self._recommender_name_list = [None] * self._recommender_iterator.len()

        for index, (recommender_instance, recommender_name) in enumerate(self._recommender_iterator):
            print("CarouselLayoutOptimizer_IncrementalGreedy: Model {}/{}".format(index + 1, self._recommender_iterator.len()))

            recommender_instance.load_model(self._model_folder, file_name=recommender_name + "_best_model.zip")
            self._recommender_instance_list[index] = recommender_instance
            self._recommender_name_list[index] = recommender_name
            # result_dict, _ = self._evaluator_validation.evaluateRecommender(recommender_instance)

            if recommender_instance not in carousel_list:
                evaluator_validation = EvaluatorMultipleCarousels(self._URM_validation,
                                                                  cutoff_list = [self._cutoff],
                                                                  exclude_seen = True,
                                                                  carousel_recommender_list = carousel_list)

                result_dict, _ = evaluator_validation.evaluateRecommender(recommender_instance)

                cutoff = list(result_dict.keys())[0]
                self._recommender_result[index] = result_dict[cutoff][self._metric_to_optimize]

            else:
                self._recommender_result[index] = -np.inf


    def get_layout(self, n_carousels):
        # Sort models
        result_carousel_list = []
        result_recommender_name_list = []

        for index in range(0, n_carousels):

            print("CarouselLayoutOptimizer_IncrementalGreedy: Carousel {}/{}".format(index + 1, n_carousels))

            self._evaluate_models(result_carousel_list)

            recommender_argsort = np.argsort(-self._recommender_result)
            selected = recommender_argsort[0]

            result_carousel_list.append(self._recommender_instance_list[selected])
            result_recommender_name_list.append(self._recommender_name_list[selected])

        return result_carousel_list, result_recommender_name_list



