from anndata import AnnData
import scanpy as sc
import numpy as np
import time
from memory_profiler import memory_usage
import os
import pickle 
import gc

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin

from FACSPy.model._sampling import GateSampler
from typing import Optional

from ..utils.classifier_scoring import SCORES_TO_USE, score_classifier, write_to_scores
from ..utils.classifier_list import CLASSIFIERS_TO_TEST
from ..utils.hyperparameter_search import conduct_hyperparameter_search
from ..utils._utils import _has_been_analyzed

from ._utils import SCALER, SAMPLER_KWARGS

from scipy.sparse import csr_matrix

def _generate_parameters_for_analysis_check(algorithm: str,
                                            sampling: bool,
                                            train_size: int,
                                            tuned: bool,
                                            sample_IDs: list[str],
                                            gates: list[str]) -> list[str]:
    algorithm = [str(algorithm)]
    sampling = [str(sampling)]
    train_size = [str(train_size)]
    tuned = [str(param) for param in tuned]

    sample_IDs = list(sample_IDs)
    gates = [gate for gate in gates]
    score_on = ["train", "test", "val"]

    return [algorithm, sampling, score_on, train_size, tuned, sample_IDs, gates]

def _get_classifier(classifier_name,
                    params: Optional[dict] = None,
                    hyperparameter: bool = False) -> ClassifierMixin:
    if params is None:
        params = {}
    if classifier_name in ["RandomForestClassifier", "ExtraTreesClassifier"] and not hyperparameter:
        params["n_jobs"] = 16
    return CLASSIFIERS_TO_TEST[classifier_name]["classifier"](**params)

def _run_scaler_comparison(dataset: AnnData,
                           output_dir: str,
                           classifier: str,
                           continue_analysis: bool = False) -> None:
    TEST_SIZE = 0.1
    gates = dataset.uns["gating_cols"]

    scores = ",".join([score for score in SCORES_TO_USE])
    resource_metrics = "algorithm,sampling,score_on,train_size,scaler,sample_ID,gate," + scores + "\n"
    score_key = f"Scores"
    if continue_analysis is False or not os.path.isfile(os.path.join(output_dir, f"{score_key}.log")):
        write_to_scores(resource_metrics,
                        output_dir = output_dir,
                        key = score_key,
                        init = True)

    technicals = ",".join(["train_time", "pred_time_train", "pred_time_test", "pred_time_val", "min_mem", "mean_mem", "max_mem"])
    resource_metrics = "algorithm,sampling,train_size,scaler,sample_ID," + technicals + "\n"
    score_key = f"Technicals"
    if continue_analysis is False or not os.path.isfile(os.path.join(output_dir, f"{score_key}.log")):
        write_to_scores(resource_metrics,
                        output_dir = output_dir,
                        key = score_key,
                        init = True)

    samples = dataset.obs["sample_ID"].unique().tolist()

    for sampling in [True, False]:
        for train_size in [5000, 50_000, 500_000]:
            for scaling in ["MinMaxScaler", "StandardScaler"]:
                if continue_analysis is True:
                    parameters = _generate_parameters_for_analysis_check(algorithm = classifier,
                                                                         sampling = sampling,
                                                                         train_size = train_size,
                                                                         tuned = [True, False],
                                                                         sample_IDs = samples,
                                                                         gates = gates)
                    score_key = "Scores"
                    if _has_been_analyzed(os.path.join(output_dir, f"{score_key}.log"),
                                          parameters = parameters):
                        print(f"Skipping combination {parameters}")
                        continue

                for sample in samples:
                    clf = _get_classifier(classifier_name = classifier,
                                          params = None)

                    test_adata = dataset[dataset.obs["sample_ID"] == sample,:]
                    test_gating = test_adata.obsm["gating"].toarray()

                    train_adata = dataset[dataset.obs["sample_ID"] != sample,:]
                    if scaling == "MinMaxScaler":
                        scaler = MinMaxScaler()
                    elif scaling == "StandardScaler":
                        scaler = StandardScaler()

                    train_adata.layers["preprocessed"] = scaler.fit_transform(train_adata.layers["transformed"])

                    if sampling:
                        sampler = GateSampler(target_size = train_size,
                                              **SAMPLER_KWARGS)
                        X = train_adata.layers["preprocessed"]
                        y = train_adata.obsm["gating"].toarray()
                        X, y = sampler.fit_resample(X, y)

                    else:
                        sc.pp.subsample(train_adata, n_obs = train_size)
                        X = train_adata.layers["preprocessed"]
                        y = train_adata.obsm["gating"].toarray()

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_SIZE)
                    X_val, y_val = scaler.transform(test_adata.layers["transformed"]), test_gating

                    _run_classifier(clf = clf,
                                    scaler = scaler,
                                    X_train = X_train,
                                    y_train = y_train,
                                    X_test = X_test,
                                    y_test = y_test,
                                    X_val = X_val,
                                    y_val = y_val,
                                    output_dir = output_dir,
                                    train_size = train_size,
                                    sample = sample,
                                    gates = gates,
                                    classifier = classifier,
                                    sampling = sampling)
                    del clf
                    del X_train, X_test, X_val, y_train, y_test, y_val
                    del train_adata, test_adata, test_gating
                    gc.collect()
                gc.collect()
        gc.collect()
    gc.collect()

def _run_classifier(clf: RandomForestClassifier,
                    scaler: str,
                    X_train: np.ndarray,
                    y_train: np.ndarray,
                    X_test: np.ndarray,
                    y_test: np.ndarray,
                    X_val: np.ndarray,
                    y_val: np.ndarray,
                    output_dir: str,
                    train_size: int,
                    sample: str,
                    gates: list[str],
                    classifier: str,
                    sampling: bool):
    start = time.time()
    mem_usage = memory_usage((clf.fit, (X_train, y_train)), interval = 0.01)
    training_time = time.time() - start

    min_mem = np.min(mem_usage)
    mean_mem = np.mean(mem_usage)
    max_mem = np.max(mem_usage)

    start = time.time()
    train_pred = clf.predict(X_train)
    prediction_time_train = time.time() - start

    start = time.time()
    test_pred = clf.predict(X_test)
    prediction_time_pred = time.time() - start

    start = time.time()
    val_pred = clf.predict(X_val)
    prediction_time_val = time.time() - start

    for i, gate in enumerate(gates):
        scores = score_classifier(y_test = y_train[:,i],
                                  test_pred = train_pred[:,i])
        score_string = ",".join(scores)
        write_to_scores(f"{classifier},{str(sampling)},train,{train_size},{str(scaler)},{sample},{gate},{score_string}",
                        output_dir = output_dir,
                        key = "Scores")
        scores = score_classifier(y_test = y_test[:,i],
                                  test_pred = test_pred[:,i])
        score_string = ",".join(scores)
        write_to_scores(f"{classifier},{str(sampling)},test,{train_size},{str(scaler)},{sample},{gate},{score_string}",
                        output_dir = output_dir,
                        key = "Scores")
        scores = score_classifier(y_test = y_val[:,i],
                                  test_pred = val_pred[:,i])
        score_string = ",".join(scores)
        write_to_scores(f"{classifier},{str(sampling)},val,{train_size},{str(scaler)},{sample},{gate},{score_string}",
                        output_dir = output_dir,
                        key = "Scores")

    technicals = [training_time, prediction_time_train, prediction_time_pred, prediction_time_val, min_mem, mean_mem, max_mem]
    technicals = [str(entry) for entry in technicals]
    technicals_string = ",".join(technicals + ["\n"])
    write_to_scores(f"{classifier},{str(sampling)},{train_size},{str(scaler)},{sample},{technicals_string}",
                    output_dir = output_dir,
                    key = "Technicals")
    
    return


def run():
    from ..utils._mouse_lineages_utils import _create_dataset
    ORGAN = "bm"
    INPUT_DIR = "../datasets/mouse_lineages/"
    OUTPUT_DIR = f"../figure_data/Figure_5/mouse_lineages_{ORGAN}/sampler_DT/"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    dataset = _create_dataset(input_directory = INPUT_DIR,
                              organ = ORGAN)

    _run_scaler_comparison(dataset = dataset,
                           classifier = "DecisionTreeClassifier",
                           output_dir = OUTPUT_DIR)