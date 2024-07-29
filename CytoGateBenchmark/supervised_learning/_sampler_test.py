import os
from anndata import AnnData
import numpy as np
import scanpy as sc
import gc 
from FACSPy.model._sampling import GateSampler

from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin

from memory_profiler import memory_usage

import time

from ._utils import SCALER
from ._hyperparameter_tuning import _run_classifier

from ..utils.classifier_list import CLASSIFIERS_TO_TEST
from ..utils.classifier_scoring import SCORES_TO_USE, write_to_scores, score_classifier
from ..utils._utils import _has_been_analyzed

def _generate_parameters_for_analysis_check(algorithm: str,
                                            oversampler: str,
                                            oversample_rare_cells: bool,
                                            rare_cell_cutoff: int,
                                            rare_cells_target_fraction: float,
                                            sample_IDs: str,
                                            gates: list[str]) -> list[str]:
    algorithm = [str(algorithm)]
    oversampler = [str(oversampler)]
    oversample_rare_cells = [str(oversample_rare_cells)]
    rare_cell_cutoff = [str(rare_cell_cutoff)]
    rare_cells_target_fraction = [str(rare_cells_target_fraction)]
    sample_IDs = list(sample_IDs)
    gates = [gate for gate in gates]
    score_on = ["train", "test", "val"]

    return [algorithm, oversampler, oversample_rare_cells, rare_cell_cutoff, rare_cells_target_fraction, score_on, sample_IDs, gates]


PARAMS = {
     "RandomForestClassifier": {"n_jobs": 16},
     "DecisionTreeClassifier": {},
     "ExtraTreesClassifier": {"n_jobs": 16},
     "ExtraTreeClassifier": {},
     "KNN": {},
     "MLPClassifier": {}
}

def _run_sampler_test(dataset: AnnData,
                      output_dir: str,
                      classifier: str,
                      continue_analysis: bool = False):
    TEST_SIZE = 0.1    
    gates = dataset.uns["gating_cols"]
    scores = ",".join([score for score in SCORES_TO_USE])
    print("continue analysis is ", continue_analysis)
    resource_metrics = "algorithm,oversampler,oversample_rare_cells,rare_cell_cutoff,rare_cells_target_fraction,score_on,sample_ID,gate," + scores + "\n"
    score_key = f"Scores"
    if continue_analysis is False or not os.path.isfile(os.path.join(output_dir, f"{score_key}.log")):
        write_to_scores(resource_metrics,
                        output_dir = output_dir,
                        key = score_key,
                        init = True)
    technicals = ",".join(["train_time", "pred_time_train", "pred_time_test", "pred_time_val", "min_mem", "mean_mem", "max_mem"])
    resource_metrics = "algorithm,oversampler,oversample_rare_cells,rare_cell_cutoff,rare_cells_target_fraction,sample_ID," + technicals + "\n"
    score_key = f"Technicals"
    if continue_analysis is False or not os.path.isfile(os.path.join(output_dir, f"{score_key}.log")):
        write_to_scores(resource_metrics,
                        output_dir = output_dir,
                        key = score_key,
                        init = True)
    samples = dataset.obs["sample_ID"].unique().tolist()

    TRAIN_SIZE = 50_000

    for oversampler in ["Gaussian", "SMOTE", None]:
        for oversample_rare_cells in [False, True]:
            for rare_cell_cutoff in [10, 100, 1000, 10_000, 100_000]:
                for rare_cells_target_fraction in [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]:
                    ## we dont need to run again if we set oversample_rare_cells to False
                    ## or if there is no oversampling...
                    if oversampler is None and rare_cells_target_fraction != 0.001:
                        continue
                    if oversample_rare_cells == False and rare_cells_target_fraction != 0.001:
                        continue
                    if continue_analysis is True:
                        parameters = _generate_parameters_for_analysis_check(algorithm = classifier,
                                                                             oversampler = oversampler,
                                                                             oversample_rare_cells = oversample_rare_cells,
                                                                             rare_cell_cutoff = rare_cell_cutoff,
                                                                             rare_cells_target_fraction = rare_cells_target_fraction,
                                                                             sample_IDs = samples,
                                                                             gates = gates)
                        score_key = "Scores"
                        if _has_been_analyzed(os.path.join(output_dir, f"{score_key}.log"),
                                              parameters = parameters):
                            print(f"Skipping combination {parameters}")
                            continue
                    for sample in samples:
                        test_adata = dataset[dataset.obs["sample_ID"] == sample,:]
                        test_gating = test_adata.obsm["gating"].toarray()

                        train_adata = dataset[dataset.obs["sample_ID"] != sample,:]

                        scaler = SCALER
                        scaler.fit(train_adata.layers["transformed"])
                        train_adata.layers["preprocessed"] = scaler.transform(train_adata.layers["transformed"])

                        if oversampler is not None:
                            sampler = GateSampler(oversampler = oversampler,
                                                  target_size = TRAIN_SIZE,
                                                  target_size_per_gate = None,
                                                  oversample_rare_cells = oversample_rare_cells,
                                                  rare_cells_cutoff = rare_cell_cutoff,
                                                  rare_cells_target_size_per_gate = None,
                                                  rare_cells_target_fraction = rare_cells_target_fraction)
                            X = train_adata.layers["preprocessed"]
                            y = train_adata.obsm["gating"].toarray()
                            X, y = sampler.fit_resample(X, y)
                        else:
                            sc.pp.subsample(train_adata, n_obs = TRAIN_SIZE)
                            X = train_adata.layers["preprocessed"]
                            y = train_adata.obsm["gating"].toarray()
                        
                        print("X shape: ", X.shape, "y shape: ", y.shape)

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_SIZE)
                        X_val, y_val = scaler.transform(test_adata.layers["transformed"]), test_gating

                        params = PARAMS[classifier]
                        clf = CLASSIFIERS_TO_TEST[classifier]["classifier"](**params)

                        _run_classifier(clf = clf,
                                        oversampler = oversampler,
                                        oversample_rare_cells = oversample_rare_cells,
                                        rare_cell_cutoff = rare_cell_cutoff,
                                        rare_cells_target_fraction = rare_cells_target_fraction,
                                        X_train = X_train,
                                        y_train = y_train,
                                        X_test = X_test,
                                        y_test = y_test,
                                        X_val = X_val,
                                        y_val = y_val,
                                        output_dir = output_dir,
                                        sample = sample,
                                        gates = gates,
                                        classifier = classifier)
                        
                        del X, y, X_train, X_test, X_val, y_train, y_test, y_val, clf
                        del test_adata, train_adata, test_gating, scaler
                        gc.collect()
                    gc.collect()
                gc.collect()
            gc.collect()
        gc.collect()
    return

def _run_classifier(clf: ClassifierMixin,
                    oversampler: str,
                    oversample_rare_cells: bool,
                    rare_cell_cutoff: float,
                    rare_cells_target_fraction: float,
                    X_train: np.ndarray,
                    y_train: np.ndarray,
                    X_test: np.ndarray,
                    y_test: np.ndarray,
                    X_val: np.ndarray,
                    y_val: np.ndarray,
                    output_dir: str,
                    sample: str,
                    gates: list[str],
                    classifier: str):
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
        write_to_scores(f"{classifier},{oversampler},{str(oversample_rare_cells)},{str(rare_cell_cutoff)},{str(rare_cells_target_fraction)},train,{sample},{gate},{score_string}",
                        output_dir = output_dir,
                        key = "Scores")
        scores = score_classifier(y_test = y_test[:,i],
                                  test_pred = test_pred[:,i])
        score_string = ",".join(scores)
        write_to_scores(f"{classifier},{oversampler},{str(oversample_rare_cells)},{str(rare_cell_cutoff)},{str(rare_cells_target_fraction)},test,{sample},{gate},{score_string}",
                        output_dir = output_dir,
                        key = "Scores")
        scores = score_classifier(y_test = y_val[:,i],
                                  test_pred = val_pred[:,i])
        score_string = ",".join(scores)
        write_to_scores(f"{classifier},{oversampler},{str(oversample_rare_cells)},{str(rare_cell_cutoff)},{str(rare_cells_target_fraction)},val,{sample},{gate},{score_string}",
                        output_dir = output_dir,
                        key = "Scores")

    technicals = [training_time, prediction_time_train, prediction_time_pred, prediction_time_val, min_mem, mean_mem, max_mem]
    technicals = [str(entry) for entry in technicals]
    technicals_string = ",".join(technicals + ["\n"])
    write_to_scores(f"{classifier},{oversampler},{str(oversample_rare_cells)},{str(rare_cell_cutoff)},{str(rare_cells_target_fraction)},{sample},{technicals_string}",
                    output_dir = output_dir,
                    key = "Technicals")
    
    return