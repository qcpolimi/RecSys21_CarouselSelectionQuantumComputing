# Optimizing the Selection of Recommendation Carousels with Quantum Computing

This is the repository with the code for the paper "Optimizing the Selection of Recommendation Carousels with Quantum Computing", published at RecSys Late-Breaking Results 2021. 
Developed by **[Maurizio Ferrari Dacrema](https://github.com/maurizioFD)** and **[Nicolò Felicioni](https://github.com/nicolo-felicioni)**, at Politecnico di Milano.

See the websites of our [Quantum Computing Group](https://quantum.polimi.it/) and of our [Recommender Systems Group](http://recsys.deib.polimi.it/) for more information on our team and research activities.

For information on the requirements and how to install this repository, see the following [Installation](#Installation) section, for information on the structure of the repo and the recommender models see the [Project structure](#Project-structure) section.
Instructions on how to run the experiments are in section [Run the experiments](#Run-the-experiments). The experiments will run by default locally with a Simulated Annealing solver but it is possible to use the D-Wave quantum annealer. Further details on how to setup the quantum annealer are reported in the [D-Wave Setup](#D-Wave-Setup) section


## Installation

Note that this repository requires Python 3.8

First we suggest you create an environment for this project using Anaconda.

Checkout this repository, then enter in the repository folder and run this commands to create and activate a new environment:
```console
conda create -n CarouselEvaluation python=3.8 anaconda
conda activate CarouselEvaluation
```

In order to compile you must have installed: _gcc_ and _python3 dev_, which can be installed with the following commands:
```console
sudo apt install gcc 
sudo apt-get install python3-dev
```

Then install all the requirements and dependencies
```console
pip install -r requirements.txt
```

At this point you can compile all Cython algorithms by running the following command. The script will compile within the current active environment. The code has been developed for Linux and Windows platforms. During the compilation you may see some warnings. 
 
```console
python run_compile_all_cython.py
```


### D-Wave Setup

In order to use the D-Wave quantum annealer via their cloud services you must first sign-up to [D-Wave Leap](https://cloud.dwavesys.com/leap/)
and get your API token. Then, you need to run the following command in the project Python environment:

```console
dwave setup
```

This is a guided setup for D-Wave Ocean SDK. When asked to select non-open-source packages to install you should
answer `y` and install at least _D-Wave Drivers_ (the D-Wave Problem Inspector package is not required, but could be
useful to analyse problem solutions, if solving problems with the QPU only).

Then, continue the configuration by setting custom properties (or keeping the default ones, as we suggest), apart from
the `Authentication token` field, where you should paste your API token obtained on the D-Wave Leap dashboard.

You should now be able to connect to D-Wave cloud services. In order to verify the connection, you can use the following
command, which will send a test problem to D-Wave's QPU:

```console
dwave ping
```


## Run the experiments

See see the following [Installation](#Installation) section for information on how to install this repository.
After the installation is complete you can run the experiments.

* Run the 'run_hyperparameter_optimization.py' script to perform the hyperparameter optimization of the various algorithms independently. In the script you may select which of the datasets to use. The script will automatically download the data (only possible for MovieLens10M. For the Netflix dataset you will have to download it from the link that will be prompted) and save the optimized models.
* Run the 'run_layout_optimization_QUBO.py' to run all experiments reported in the paper. The default will run the QUBO selection algorithm with a classical SA solver locally, add the "--use_QPU True" flag if you wish to solve them on the QPU. We suggest you to be careful as it could use up most, if not all, of your monthly free QPU quota. If you wish to run the exact search for an optimal global selection use the "--exact_search True" flag. This is very computationally expensive and will likely require more than a day for more than 4 carousels. 
The script will create several csv files that contain the results for each setting. The results for the carousel selection will be saved in folder "QUBO_SA_alpha" for each target number of carousels and QUBO model. Lastly, in the same folder you will find files "Summary_table_PRECISION.csv" that will summarize the precision value of all selection strategies.




## Project structure

#### Evaluation
The Evaluator class is used to evaluate a recommender object. It computes various metrics:
* Accuracy metrics: ROC_AUC, PRECISION, RECALL, MAP, MRR, NDCG, F1, HIT_RATE, ARHR
* Beyond-accuracy metrics: NOVELTY, DIVERSITY, COVERAGE

The evaluator takes as input the URM against which you want to test the recommender, then a list of cutoff values (e.g., 5, 20) and, if necessary, an object to compute diversity.
The function evaluateRecommender will take as input only the recommender object you want to evaluate and return both a dictionary in the form {cutoff: results}, where results is {metric: value} and a well-formatted printable string.

```python

    from Evaluation.Evaluator import EvaluatorHoldout

    evaluator_test = EvaluatorHoldout(URM_test, [5, 20])

    results_run_dict, results_run_string = evaluator_test.evaluateRecommender(recommender_instance)

    print(results_run_string)

```


### Recommenders
Contains some basic modules and the base classes for different Recommender types.
All recommenders inherit from BaseRecommender, therefore have the same interface.
You must provide the data when instantiating the recommender and then call the _fit_ function to build the corresponding model.

Each recommender has a _compute_item_score function which, given an array of user_id, computes the prediction or _score_ for all items.
Further operations like removing seen items and computing the recommendation list of the desired length are done by the _recommend_ function of BaseRecommender

As an example:

```python
    user_id = 158
    
    recommender_instance = ItemKNNCFRecommender(URM_train)
    recommender_instance.fit(topK=150)
    recommended_items = recommender_instance.recommend(user_id, cutoff = 20, remove_seen_flag=True)
    
    recommender_instance = SLIM_ElasticNet(URM_train)
    recommender_instance.fit(topK=150, l1_ratio=0.1, alpha = 1.0)
    recommended_items = recommender_instance.recommend(user_id, cutoff = 20, remove_seen_flag=True)
```

The similarity module allows to compute the item-item or user-user similarity.
It is used by calling the Compute_Similarity class and passing which is the desired similarity and the sparse matrix you wish to use.

It is able to compute the following similarities: Cosine, Adjusted Cosine, Jaccard, Tanimoto, Pearson and Euclidean (linear and exponential)

```python
    similarity = Compute_Similarity(URM_train, shrink=shrink, topK=topK, normalize=normalize, similarity = "cosine")

    W_sparse = similarity.compute_similarity()

```


### Data Reader and splitter
DataReader objects read the dataset from its original file and save it as a sparse matrix.

DataSplitter objects take as input a DataReader and split the corresponding dataset in the chosen way.
At each step the data is automatically saved in a folder, though it is possible to prevent this by setting _save_folder_path = False_ when calling _load_data_.
If a DataReader or DataSplitter is called for a dataset which was already processed, the saved data is loaded.

DataPostprocessing can also be applied between the dataReader and the dataSplitter and nested in one another.

When you have bilt the desired combination of dataset/preprocessing/split, get the data calling _load_data_.

```python
dataset = Movielens1MReader()

dataset = DataPostprocessing_K_Cores(dataset, k_cores_value=25)
dataset = DataPostprocessing_User_sample(dataset, user_quota=0.3)
dataset = DataPostprocessing_Implicit_URM(dataset)

dataSplitter = DataSplitter_Warm_k_fold(dataset)

dataSplitter.load_data()

URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()
```




