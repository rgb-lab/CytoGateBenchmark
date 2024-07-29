import os

from anndata import AnnData
import FACSPy as fp
import pandas as pd
import numpy as np
import time
import gc
from memory_profiler import memory_usage

from ..utils.classifier_scoring import (SCORES_TO_USE,
                                        write_to_scores,
                                        score_classifier)
from ..utils._utils import _has_been_analyzed

def _generate_parameters_for_analysis_check(algorithm: str,
                                            sample_IDs: list[str],
                                            gate_mapping: dict,
                                            full_gate_provided: bool,
                                            dataset: AnnData,
                                            technicals: bool = False):
    if algorithm == "flowsom":
        algorithm = ["flowsom_fixed_clusters", "flowsom_non_fixed_clusters"]
    else:
        algorithm = [algorithm]

    sample_IDs = list(sample_IDs)
    
    gates = [gate for gate in gate_mapping]
    if technicals:
        return [algorithm, sample_IDs]
    return [algorithm, sample_IDs, gates]

def _score_gating(dataset: AnnData,
                  gate_mapping: dict,
                  algorithm: str,
                  output_dir: str,
                  score_key: str,
                  full_gate_provided: bool) -> None:

    gating = pd.DataFrame(data = dataset.obsm["gating"].toarray(),
                          index = dataset.obs_names,
                          columns = dataset.uns["gating_cols"])
    gating[dataset.obs.columns] = dataset.obs

    for sample in gating["sample_ID"].unique():
        print(f"Scoring sample with sample_ID: {sample}")
        gate_df: pd.DataFrame = gating[gating["sample_ID"] == sample].copy()
        for flowjo_gate in list(gate_mapping.keys()):
            if not full_gate_provided:
                full_gate = fp._utils._find_gate_path_of_gate(dataset, flowjo_gate)
            else:
                full_gate = flowjo_gate
            parent_gate = fp._utils._find_parent_gate(full_gate)
            unsup_gate = gate_mapping[flowjo_gate]
            if not full_gate_provided:
                unsup_full_gate = fp._utils._find_gate_path_of_gate(dataset, unsup_gate)
            else:
                unsup_full_gate = unsup_gate
            total_scores_parent = score_classifier(y_test = gate_df.loc[gate_df[parent_gate] == 1, full_gate].values,
                                                   test_pred = gate_df.loc[gate_df[parent_gate] == 1, unsup_full_gate].values)
            score_string = ",".join(total_scores_parent)
            write_to_scores(f"{algorithm},{sample},{flowjo_gate},{score_string}",
                            output_dir = output_dir,
                            key = score_key)
    return

def _run_cluster_algorithm_comparison(dataset: AnnData,
                                      output_dir: str,
                                      gating_strategy: dict,
                                      gate_mapping: dict,
                                      full_gate_provided: bool = False,
                                      continue_analysis: bool = False):
    safety = dataset.copy()

    # first, we test the algorithms without hyperparameter tuning
    # to assess the accuracy of the gating results.
    # for flowsom, we have two tests: one with agglomerative clustering
    # and one with a fixed cluster size of 30

    # current experiment is just for algorithm comparison
    scores = ",".join([score for score in SCORES_TO_USE])
    resource_metrics = "algorithm,sample_ID,gate," + scores + "\n"
    score_key = f"Scores"
    if continue_analysis is False or not os.path.isfile(os.path.join(output_dir, f"{score_key}.log")):
        write_to_scores(resource_metrics,
                        output_dir = output_dir,
                        key = score_key,
                        init = True)
        

    for algorithm in ["leiden", "parc", "flowsom", "phenograph"]:
        dataset = safety.copy()
        parameters = _generate_parameters_for_analysis_check(algorithm = algorithm,
                                                             sample_IDs = dataset.obs["sample_ID"].unique(),
                                                             gate_mapping = gate_mapping,
                                                             full_gate_provided = full_gate_provided,
                                                             dataset = dataset)
        if _has_been_analyzed(os.path.join(output_dir, f"{score_key}.log"),
                              parameters = parameters):
            print(f"Skipping combination {parameters}")
            continue

        clf = fp.ml.unsupervisedGating(dataset,
                                       gating_strategy = gating_strategy,
                                       clustering_algorithm = algorithm,
                                       layer = "transformed")
        if algorithm == "flowsom":
            try:
                clf.identify_populations()
            except Exception as e:
                with open(os.path.join(output_dir, "Errors.log"), "a") as error_log:
                    error_log.write("An Error occured\n")
                    error_log.write(str(e))
                    error_log.write("\n")
                    continue
        else:
            clf.identify_populations()
        _score_gating(dataset,
                      gate_mapping = gate_mapping,
                      algorithm = algorithm,
                      output_dir = output_dir,
                      score_key = score_key,
                      full_gate_provided = full_gate_provided)
        gc.collect()

    # next, we test the metrics time and memory consumption
    # we repeat the analysis as we want a per sample analysis
    samples = dataset.obs["sample_ID"].unique()
    resource_metrics = "algorithm,sample_ID,train_time,min_mem,max_mem,mean_mem\n"
    score_key = "Technicals"
    if continue_analysis is False or not os.path.isfile(os.path.join(output_dir, f"{score_key}.log")):
        write_to_scores(resource_metrics,
                        output_dir = output_dir,
                        key = score_key,
                        init = True)
    for algorithm in ["leiden", "parc", "flowsom", "phenograph"]:
        dataset = safety.copy()
        for sample in samples:
            parameters = _generate_parameters_for_analysis_check(algorithm = algorithm,
                                                                 sample_IDs = [sample],
                                                                 gate_mapping = gate_mapping,
                                                                 full_gate_provided = full_gate_provided,
                                                                 dataset = dataset,
                                                                 technicals = True)
            if _has_been_analyzed(os.path.join(output_dir, f"{score_key}.log"),
                                  parameters = parameters):
                print(f"Skipping combination {parameters}")
                continue

            data_subset = dataset[dataset.obs["sample_ID"] == sample].copy()
            start = time.time()
            clf = fp.ml.unsupervisedGating(data_subset,
                                           gating_strategy = gating_strategy,
                                           clustering_algorithm = algorithm,
                                           layer = "transformed")
            if algorithm == "flowsom":
                try:
                    mem_usage = memory_usage((clf.identify_populations, ()), interval = 10)
                except Exception as e:
                    with open(os.path.join(output_dir, "Errors.log"), "a") as error_log:
                        error_log.write("An Error occured\n")
                        error_log.write(str(e))
                        error_log.write("\n")
                        continue
            else:
                mem_usage = memory_usage((clf.identify_populations, ()), interval = 10)
            stop = time.time() - start
            metric_string = f"{algorithm},{sample},{stop},{np.min(mem_usage)},{np.mean(mem_usage)},{np.max(mem_usage)}\n"
            write_to_scores(metric_string,
                            output_dir = output_dir,
                            key = score_key)
            gc.collect()

    return
