import os

import FACSPy as fp
import pandas as pd
from anndata import AnnData
import numpy as np

from ..utils.classifier_scoring import SCORES_TO_USE, write_to_scores, score_classifier

import time
import gc

from ..utils._utils import _has_been_analyzed
from memory_profiler import memory_usage

def _generate_parameters_for_analysis_check(algorithm: str,
                                            sample_IDs: list[str],
                                            gate_mapping: dict,
                                            n_iter: int,
                                            resolution: int,
                                            full_gate_provided: bool,
                                            dataset: AnnData,
                                            technicals: bool = False):
    resolution = [str(resolution)]
    n_iter = [str(n_iter)]
    algorithm = [str(algorithm)]
    
    sample_IDs = list(sample_IDs)

    gates = [gate for gate in gate_mapping]

    if technicals:
        return [algorithm, sample_IDs, n_iter, resolution]

    return [algorithm, sample_IDs, gates, n_iter, resolution]


def _score_gating(dataset: AnnData,
                  gate_mapping: dict,
                  resolution: str,
                  n_iter_leiden: int,
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
            write_to_scores(f"parc,{sample},{flowjo_gate},{n_iter_leiden},{resolution},{score_string}",
                            output_dir = output_dir,
                            key = score_key)
    return

def _run_parc_benchmark(dataset: AnnData,
                          output_dir: str,
                          gating_strategy: dict,
                          gate_mapping: dict,
                          full_gate_provided: bool = False,
                          continue_analysis: bool = False):
    safety = dataset.copy()

    scores = ",".join([score for score in SCORES_TO_USE])
    resource_metrics = "algorithm,sample_ID,gate,n_iter_leiden,resolution," + scores + "\n"
    score_key = f"Scores"
    if continue_analysis is False or not os.path.isfile(os.path.join(output_dir, f"{score_key}.log")):
        write_to_scores(resource_metrics,
                        output_dir = output_dir,
                        key = score_key,
                        init = True)
    resolutions = [0.1, 0.5, 1, 5, 10, 15, 20, 40, 60]
    n_iters = [1,3,5,10,20,30,50]
    for resolution in resolutions:
        for n_iter in n_iters:
            parameters = _generate_parameters_for_analysis_check(algorithm = "parc",
                                                                 sample_IDs = dataset.obs["sample_ID"].unique(),
                                                                 gate_mapping = gate_mapping,
                                                                 resolution = resolution,
                                                                 n_iter = n_iter,
                                                                 full_gate_provided = full_gate_provided,
                                                                 dataset = dataset)
            if _has_been_analyzed(os.path.join(output_dir, f"{score_key}.log"),
                                  parameters = parameters):
                print(f"Skipping combination {parameters}")
                continue

            dataset = safety.copy()
            clf = fp.ml.unsupervisedGating(dataset,
                                           gating_strategy = gating_strategy,
                                           clustering_algorithm = "parc",
                                           layer = "transformed")
            clf.identify_populations(cluster_kwargs = {"resolution_parameter": resolution,
                                                       "n_iter_leiden": n_iter,
                                                       "small_pop": 10})
            _score_gating(dataset,
                          gate_mapping = gate_mapping,
                          resolution = resolution,
                          n_iter_leiden = n_iter,
                          output_dir = output_dir,
                          score_key = score_key,
                          full_gate_provided = full_gate_provided)
            gc.collect()

    # next, we test the metrics time and memory consumption
    # we repeat the analysis as we want a per sample analysis
    samples = dataset.obs["sample_ID"].unique()
    resource_metrics = "algorithm,sample_ID,n_iter_leiden,resolution,train_time,min_mem,max_mem,mean_mem\n"
    score_key = f"Technicals"
    if continue_analysis is False or not os.path.isfile(os.path.join(output_dir, f"{score_key}.log")):
        write_to_scores(resource_metrics,
                        output_dir = output_dir,
                        key = score_key,
                        init = True)
    for resolution in resolutions:
        for n_iter in n_iters:
            dataset = safety.copy()
            for sample in samples:
                parameters = _generate_parameters_for_analysis_check(algorithm = "parc",
                                                                     sample_IDs = [sample],
                                                                     gate_mapping = gate_mapping,
                                                                     resolution = resolution,
                                                                     n_iter = n_iter,
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
                                               clustering_algorithm = "parc",
                                               layer = "transformed")
                mem_usage = memory_usage((clf.identify_populations, (), {"cluster_kwargs": {"resolution_parameter": resolution,
                                                                                            "n_iter_leiden": n_iter,
                                                                                            "small_pop": 10}}), interval = 10)
                stop = time.time() - start
                metric_string = f"parc,{sample},{n_iter},{resolution},{stop},{np.min(mem_usage)},{np.mean(mem_usage)},{np.max(mem_usage)}\n"
                write_to_scores(metric_string,
                                output_dir = output_dir,
                                key = score_key)
                gc.collect()

    return
