#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/02/2021

@author: anonymous for blind review
"""

from dwave.system.composites import LazyFixedEmbeddingComposite
from Recommenders.DataIO import DataIO
import numpy as np
import dimod, itertools
import pandas as pd
from dwave.system import DWaveCliqueSampler
from scipy.stats import pearsonr, kendalltau, spearmanr
import time

def model_selection(BQM, recommender_iterator, num_reads, n_carousels, sampler):

    start_time = time.time()
    response = sampler.sample(BQM, num_reads=num_reads)
    client_time = time.time() - start_time

    recommender_name_list = [recommender_name for _,recommender_name in recommender_iterator]
    result_dataframe = pd.DataFrame()


    for read in response.data(['sample', 'energy', 'num_occurrences']):

        result_dict = {
            **{"variable_{}".format(recommender_name): read.sample[recommender_name]>0.0 for recommender_name in recommender_name_list},
            "n_selected": sum(read.sample[name] for name in recommender_name_list),
            "energy": read.energy,
            "num_occurrences": read.num_occurrences,
        }

        result_dataframe = result_dataframe.append(result_dict, ignore_index=True)

    result_dataframe["valid"] = result_dataframe["n_selected"] == n_carousels

    # Remove duplicates
    result_dataframe_nodup = result_dataframe.groupby(result_dataframe.columns.tolist()).size().reset_index().rename(columns={0: 'occurrence'})

    assert len(result_dataframe) == result_dataframe_nodup["occurrence"].sum()

    print("Optimizaton complete. Valid results are {}/{} ({:.1f}%)".format(result_dataframe["valid"].sum(), num_reads, result_dataframe["valid"].sum()/num_reads*100))

    timing = {"client_time": client_time}

    if isinstance(sampler, LazyFixedEmbeddingComposite):
        timing.update(response.info['timing'])


    return result_dataframe_nodup, timing




def maximum_energy_delta(bqm):
    """Compute conservative bound on maximum change in energy when flipping a single variable"""
    delta_max = 0
    for i in bqm.iter_variables():
        delta = abs(bqm.get_linear(i))
        for j in bqm.iter_neighbors(i):
            delta += abs(bqm.get_quadratic(i,j))
        if delta > delta_max:
            delta_max = delta
    return delta_max


class QUBOModelSelection(object):

    def __init__(self, recommender_iterator):
        super(QUBOModelSelection, self).__init__()

        self._recommender_iterator = recommender_iterator


    def get_BQM(self):

        bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)

        print("Adding Bias")
        for index, (recommender_instance, recommender_name) in enumerate(self._recommender_iterator):
            # data_dict = dataIO.load_data(recommender_name)
            bias = self.get_variable_bias(recommender_name)
            bqm.add_variable(recommender_name, bias)

        print("Adding Interactions")
        for tuple0, tuple1 in itertools.combinations(self._recommender_iterator, 2):

            recommender_instance_0, recommender_name_0 = tuple0
            recommender_instance_1, recommender_name_1 = tuple1

            if recommender_name_0 != recommender_name_1:

                interaction_0_1 = self.get_variable_interaction(recommender_name_0, recommender_name_1)

                bqm.add_interaction(recommender_name_0, recommender_name_1, interaction_0_1)
                bqm.add_interaction(recommender_name_1, recommender_name_0, interaction_0_1)


        return bqm

    def get_variable_bias(self, recommender_name):
        raise NotImplementedError

    def get_variable_interaction(self, recommender_name_0, recommender_name_1):
        raise NotImplementedError





class QUBOModelSelection_Pearson(QUBOModelSelection):

    def __init__(self, recommender_iterator, data_folder):
        super(QUBOModelSelection_Pearson, self).__init__(recommender_iterator)

        self._data_folder = data_folder
        self._dataIO = DataIO(folder_path=self._data_folder)


    def get_variable_bias(self, recommender_name):
        return 1.0

    def get_variable_interaction(self, recommender_name_0, recommender_name_1):

        data_dict_0 = self._dataIO.load_data(recommender_name_0)
        data_dict_1 = self._dataIO.load_data(recommender_name_1)

        correlation, _ = pearsonr (data_dict_0["item_recommendation_count"],
                                   data_dict_1["item_recommendation_count"])

        return correlation




class QUBOModelSelection_Kendall(QUBOModelSelection):

    def __init__(self, recommender_iterator, data_folder):
        super(QUBOModelSelection_Kendall, self).__init__(recommender_iterator)

        self._data_folder = data_folder
        self._dataIO = DataIO(folder_path=self._data_folder)


    def get_variable_bias(self, recommender_name):
        return 1.0

    def get_variable_interaction(self, recommender_name_0, recommender_name_1):

        data_dict_0 = self._dataIO.load_data(recommender_name_0)
        data_dict_1 = self._dataIO.load_data(recommender_name_1)

        correlation, _ = kendalltau (data_dict_0["item_recommendation_count"],
                                     data_dict_1["item_recommendation_count"])

        return correlation


class QUBOModelSelection_Spearman(QUBOModelSelection):

    def __init__(self, recommender_iterator, data_folder):
        super(QUBOModelSelection_Spearman, self).__init__(recommender_iterator)

        self._data_folder = data_folder
        self._dataIO = DataIO(folder_path=self._data_folder)


    def get_variable_bias(self, recommender_name):
        return 1.0

    def get_variable_interaction(self, recommender_name_0, recommender_name_1):

        data_dict_0 = self._dataIO.load_data(recommender_name_0)
        data_dict_1 = self._dataIO.load_data(recommender_name_1)

        correlation, _ = spearmanr (data_dict_0["item_recommendation_count"],
                                    data_dict_1["item_recommendation_count"])

        return correlation


from Data_manager.Dataset import gini_index

class QUBOModelSelection_GiniDiff(QUBOModelSelection):

    def __init__(self, recommender_iterator, data_folder):
        super(QUBOModelSelection_GiniDiff, self).__init__(recommender_iterator)

        self._data_folder = data_folder
        self._dataIO = DataIO(folder_path=self._data_folder)


    def get_variable_bias(self, recommender_name):
        return 1.0

    def get_variable_interaction(self, recommender_name_0, recommender_name_1):

        data_dict_0 = self._dataIO.load_data(recommender_name_0)
        data_dict_1 = self._dataIO.load_data(recommender_name_1)

        delta = data_dict_0["item_recommendation_count"] - data_dict_1["item_recommendation_count"]

        correlation = gini_index(delta)

        return -correlation




class QUBOModelSelection_Accuracy(QUBOModelSelection_GiniDiff):

    def __init__(self, recommender_iterator, recommendations_folder, model_folder, evaluator_validation, cutoff):
        super(QUBOModelSelection_GiniDiff, self).__init__(recommender_iterator)

        self._recommendations_folder = recommendations_folder
        self._model_folder = model_folder
        self._dataIO_recommendations = DataIO(folder_path=self._recommendations_folder)
        self._dataIO_model_folder = DataIO(folder_path=self._model_folder)

        self._model_to_accuracy_dict = {}

        for recommender_instance, recommender_name in recommender_iterator:
            data_dict = self._dataIO_model_folder.load_data(recommender_name + "_metadata.zip")
            self._model_to_accuracy_dict[recommender_name] = data_dict['result_on_validation_best']["PRECISION"]


    def get_variable_bias(self, recommender_name):

        return -self._model_to_accuracy_dict[recommender_name]/max(self._model_to_accuracy_dict.values())


    def get_variable_interaction(self, recommender_name_0, recommender_name_1):

        normalized_accuracy_0 = self._model_to_accuracy_dict[recommender_name_0]/max(self._model_to_accuracy_dict.values())
        normalized_accuracy_1 = self._model_to_accuracy_dict[recommender_name_1]/max(self._model_to_accuracy_dict.values())

        return -normalized_accuracy_0 * normalized_accuracy_1


