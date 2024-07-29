"""
Script to compare the algorithms leiden, phenograph, parc and flowsom
with standard settings.

First, the accuracy is determined.
After that, the technical metrics for time needed and memory consumption
are recorded.

Uses the dataset "Mouse Lineages Bone Marrow", Flow Cytometry, 13 markers.

"""

import os

from ._gating_strategies import MOUSE_LINEAGE_GATING_STRATEGY, MOUSE_LINEAGE_GATE_MAPPING
from ..unsupervised_learning._cluster_algorithm_comparison import _run_cluster_algorithm_comparison

from ..utils._mouse_lineages_utils import _create_dataset

ORGAN = "bm"
INPUT_DIR = "../datasets/mouse_lineages/"
OUTPUT_DIR = f"../figure_data/Figure_3/mouse_lineages_{ORGAN}/algorithm_comparison/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def run(continue_analysis: bool = False):
    dataset = _create_dataset(input_directory = INPUT_DIR,
                              organ = ORGAN)
    _run_cluster_algorithm_comparison(dataset = dataset,
                                      output_dir = OUTPUT_DIR,
                                      gating_strategy = MOUSE_LINEAGE_GATING_STRATEGY,
                                      gate_mapping = MOUSE_LINEAGE_GATE_MAPPING,
                                      continue_analysis = continue_analysis)

if __name__ == "__main__":
    run()