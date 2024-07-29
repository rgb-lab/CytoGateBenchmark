import os
from anndata import AnnData
import numpy as np
import scanpy as sc

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MinMaxScaler
import gc

import time

from memory_profiler import memory_usage

from ..utils.classifier_list import CLASSIFIERS_TO_TEST_FULL
from ..utils.classifier_scoring import SCORES_TO_USE, write_to_scores, score_classifier
from ..utils._utils import _has_been_analyzed

from ._utils import SCALER, SAMPLER_KWARGS
from FACSPy.model._sampling import GateSampler

def _generate_parameters_for_analysis_check(algorithm: str,
                                            sampling: bool,
                                            sample_IDs: list[str],
                                            gates: list[str]) -> list[str]:
    algorithm = [str(algorithm)]
    sampling = [str(sampling)]

    sample_IDs = list(sample_IDs)
    gates = [gate for gate in gates]
    score_on = ["train", "test", "val"]

    return [algorithm, sampling, score_on, sample_IDs, gates]

def _run_classifier_comparison(dataset: AnnData,
                               output_dir: str,
                               continue_analysis: bool = False) -> None:
    TRAIN_SIZE = 10_000
    TEST_SIZE = 0.1

    gates = dataset.uns["gating_cols"]

    scores = ",".join([score for score in SCORES_TO_USE])
    resource_metrics = "algorithm,sampling,score_on,sample_ID,gate," + scores + "\n"
    score_key = f"Scores"
    if continue_analysis is False or not os.path.isfile(os.path.join(output_dir, f"{score_key}.log")):
        write_to_scores(resource_metrics,
                        output_dir = output_dir,
                        key = score_key,
                        init = True)
    technicals = ",".join(["train_time", "pred_time_train", "pred_time_test", "pred_time_val", "min_mem", "mean_mem", "max_mem"])
    resource_metrics = "algorithm,sampling,sample_ID," + technicals + "\n"
    score_key = f"Technicals"
    if continue_analysis is False or not os.path.isfile(os.path.join(output_dir, f"{score_key}.log")):
        write_to_scores(resource_metrics,
                        output_dir = output_dir,
                        key = score_key,
                        init = True)
    samples = dataset.obs["sample_ID"].unique().tolist()
    for sampling in [True, False]:
        for classifier in CLASSIFIERS_TO_TEST_FULL:
            print(f"running classifier {classifier}")

            if "mouse_lineages_pb" in output_dir and classifier in ["LabelSpreading", "LabelPropagation"]:
                continue

            for sample in samples:
                if continue_analysis is True:
                    parameters = _generate_parameters_for_analysis_check(algorithm = classifier,
                                                                         sampling = sampling,
                                                                         sample_IDs = [sample],
                                                                         gates = gates)

                    score_key = "Scores"
                    if _has_been_analyzed(os.path.join(output_dir, f"{score_key}.log"),
                                          parameters = parameters):
                        print(f"Skipping combination {parameters}")
                        continue
                #we exclude classifiers that are not suitable for multi-input because single-input with empty gates does not work
                if not CLASSIFIERS_TO_TEST_FULL[classifier]["accepts_empty_class"]:
                    continue
                if CLASSIFIERS_TO_TEST_FULL[classifier]["allows_multi_class"]:
                    if CLASSIFIERS_TO_TEST_FULL[classifier]["multiprocessing"]:
                        clf = CLASSIFIERS_TO_TEST_FULL[classifier]["classifier"](n_jobs = 16)
                    else:
                        clf = CLASSIFIERS_TO_TEST_FULL[classifier]["classifier"]()
                else:
                    if CLASSIFIERS_TO_TEST_FULL[classifier]["scalable"] == False:
                        clf = MultiOutputClassifier(CLASSIFIERS_TO_TEST_FULL[classifier]["classifier"]())
                    else:
                        clf = MultiOutputClassifier(CLASSIFIERS_TO_TEST_FULL[classifier]["classifier"](), n_jobs = 16)

                test_adata = dataset[dataset.obs["sample_ID"] == sample,:]
                test_gating = test_adata.obsm["gating"].toarray()

                train_adata = dataset[dataset.obs["sample_ID"] != sample,:]
                if classifier in ["ComplementNB", "MultinomialNB"]:
                    scaler = MinMaxScaler()
                else:
                    scaler = SCALER
                train_adata.layers["preprocessed"] = scaler.fit_transform(train_adata.layers["transformed"])

                if sampling:
                    sampler = GateSampler(target_size = TRAIN_SIZE,
                                          **SAMPLER_KWARGS)
                    X = train_adata.layers["preprocessed"]
                    y = train_adata.obsm["gating"].toarray()
                    X, y = sampler.fit_resample(X, y)

                else:
                    sc.pp.subsample(train_adata, n_obs = TRAIN_SIZE)
                    X = train_adata.layers["preprocessed"]
                    y = train_adata.obsm["gating"].toarray()
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_SIZE)
                X_val, y_val = scaler.transform(test_adata.layers["transformed"]), test_gating

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
                    write_to_scores(f"{classifier},{str(sampling)},train,{sample},{gate},{score_string}",
                                    output_dir = output_dir,
                                    key = "Scores")
                    scores = score_classifier(y_test = y_test[:,i],
                                              test_pred = test_pred[:,i])
                    score_string = ",".join(scores)
                    write_to_scores(f"{classifier},{str(sampling)},test,{sample},{gate},{score_string}",
                                    output_dir = output_dir,
                                    key = "Scores")
                    scores = score_classifier(y_test = y_val[:,i],
                                              test_pred = val_pred[:,i])
                    score_string = ",".join(scores)
                    write_to_scores(f"{classifier},{str(sampling)},val,{sample},{gate},{score_string}",
                                    output_dir = output_dir,
                                    key = "Scores")

                technicals = [training_time, prediction_time_train, prediction_time_pred, prediction_time_val, min_mem, mean_mem, max_mem]
                technicals = [str(entry) for entry in technicals]
                technicals_string = ",".join(technicals + ["\n"])
                write_to_scores(f"{classifier},{str(sampling)},{sample},{technicals_string}",
                                output_dir = output_dir,
                                key = "Technicals")
                del clf, X_val, X_test, X_train, y_val, y_test, y_train
                del train_adata, test_adata, test_gating
                gc.collect()
            gc.collect()
        gc.collect()
    gc.collect()
            





        
        









