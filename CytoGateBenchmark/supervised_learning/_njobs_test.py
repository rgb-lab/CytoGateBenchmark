from anndata import AnnData
import scanpy as sc
import numpy as np
from ..utils._halving_random_search import HalvingRandomSearchCV_TE
import time
from memory_profiler import memory_usage
import os
import pickle 
import gc

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin

from FACSPy.model._sampling import GateSampler
from typing import Optional

from ..utils.classifier_scoring import SCORES_TO_USE, score_classifier, write_to_scores
from ..utils.classifier_list import CLASSIFIERS_TO_TEST
from ..utils.hyperparameter_search import conduct_hyperparameter_search
from ..utils._utils import _has_been_analyzed

from ._utils import _use_sampler, SCALER, SAMPLER_KWARGS

def _get_classifier(classifier_name,
                    params: Optional[dict] = None,
                    hyperparameter: bool = False) -> ClassifierMixin:
    if params is None:
        params = {}
    if classifier_name in ["RandomForestClassifier", "ExtraTreesClassifier"] and not hyperparameter:
        params["n_jobs"] = 16
    return CLASSIFIERS_TO_TEST[classifier_name]["classifier"](**params)

def _run_hyperparameter_tuning(dataset: AnnData,
                               classifier: str) -> None:

    HYPERPARAMETER_SEARCH_DATASET_SIZE = min(
        int(dataset.obs["sample_ID"].value_counts().mean() * 2),
        600_000
    )

    print(f"Running Hyperparameter search on {HYPERPARAMETER_SEARCH_DATASET_SIZE} training examples.")

    scaler = SCALER
    dataset.layers["preprocessed"] = scaler.fit_transform(dataset.layers["transformed"])

    tune_adata = sc.pp.subsample(dataset, n_obs = HYPERPARAMETER_SEARCH_DATASET_SIZE, copy = True)
    X = tune_adata.layers["preprocessed"]
    y = tune_adata.obsm["gating"].toarray()

    clf = _get_classifier(classifier_name = classifier,
                          hyperparameter = True)
    grid = CLASSIFIERS_TO_TEST[classifier]["grid"]
    for n_jobs in [8, 16, 32, 64, -1]:
        start = time.time()
        clf = _get_classifier(classifier_name = classifier,
                              hyperparameter = True)
        grid_result = HalvingRandomSearchCV_TE(estimator = clf,
                                               param_distributions = grid,
                                               scoring = "f1_macro",
                                               factor = 3,
                                               resource = "n_samples",
                                               min_resources = 1000,
                                               cv = 5,
                                               n_jobs = n_jobs,
                                               verbose = 3,
                                               error_score = 0.0,
                                               random_state = 187).fit(X, y)
        print(f"Classifier Tuning with {n_jobs} jobs completed in {time.time()-start} seconds")
        del clf, grid_result
        gc.collect()


    return

def run():
    from ..utils._mouse_lineages_utils import _create_dataset
    ORGAN = "bm"
    INPUT_DIR = "../datasets/mouse_lineages/"

    dataset = _create_dataset(input_directory = INPUT_DIR,
                              organ = ORGAN)

    _run_hyperparameter_tuning(dataset = dataset,
                               classifier = "RandomForestClassifier")