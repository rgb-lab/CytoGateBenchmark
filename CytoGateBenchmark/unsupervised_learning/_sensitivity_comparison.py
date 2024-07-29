import os

from anndata import AnnData
import FACSPy as fp
import pandas as pd
import gc

from ..utils._utils import _has_been_analyzed
from ..utils.classifier_scoring import (SCORES_TO_USE,
                                        write_to_scores,
                                        score_classifier)

def _generate_parameters_for_analysis_check(algorithm: str,
                                            sample_IDs: list[str],
                                            gate_mapping: dict,
                                            sensitivity: float,
                                            full_gate_provided: bool,
                                            dataset: AnnData):
    
    if algorithm == "flowsom":
        algorithm = ["flowsom_fixed_clusters", "flowsom_non_fixed_clusters"]
    else:
        algorithm = [str(algorithm)]
    sensitivity = [str(sensitivity)]
    
    sample_IDs = list(sample_IDs)

    gates = [gate for gate in gate_mapping]

    return [algorithm, sample_IDs, gates, sensitivity]


def _score_gating(dataset: AnnData,
                  gate_mapping: dict,
                  algorithm: str,
                  sensitivity: float,
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
            write_to_scores(f"{algorithm},{sample},{flowjo_gate},{sensitivity},{score_string}",
                            output_dir = output_dir,
                            key = score_key)
    return

def _run_sensitivity_comparison(dataset: AnnData,
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

    scores = ",".join([score for score in SCORES_TO_USE])
    resource_metrics = "algorithm,sample_ID,gate,sensitivity," + scores + "\n"
    score_key = f"Scores"
    if continue_analysis is False or not os.path.isfile(os.path.join(output_dir, f"{score_key}.log")):
        write_to_scores(resource_metrics,
                        output_dir = output_dir,
                        key = score_key,
                        init = True)

    for algorithm in ["leiden", "parc", "flowsom", "phenograph"]:
        for sensitivity in [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10, 100]:
            parameters = _generate_parameters_for_analysis_check(algorithm = algorithm,
                                                                 sample_IDs = dataset.obs["sample_ID"].unique(),
                                                                 gate_mapping = gate_mapping,
                                                                 sensitivity = sensitivity,
                                                                 full_gate_provided = full_gate_provided,
                                                                 dataset = dataset)
            if _has_been_analyzed(os.path.join(output_dir, f"{score_key}.log"),
                                  parameters = parameters):
                print(f"Skipping combination {parameters}")
                continue
            dataset = safety.copy()
            clf = fp.ml.unsupervisedGating(dataset,
                                           gating_strategy = gating_strategy,
                                           clustering_algorithm = algorithm,
                                           layer = "transformed",
                                           sensitivity = sensitivity)
            try:
                clf.identify_populations()
            except Exception as e:
                with open(os.path.join(output_dir, "Errors.log"), "a") as error_log:
                    error_log.write("An Error occured\n")
                    error_log.write(str(e))
                    error_log.write("\n")
                    continue
            _score_gating(dataset,
                          gate_mapping = gate_mapping,
                          algorithm = algorithm,
                          sensitivity = sensitivity,
                          output_dir = output_dir,
                          score_key = score_key,
                          full_gate_provided = full_gate_provided)
            gc.collect()

    return
