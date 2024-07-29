import os

from ._gating_strategies import MOUSE_LINEAGE_GATING_STRATEGY, MOUSE_LINEAGE_GATE_MAPPING
from ..unsupervised_learning._flowsom_cluster_benchmark import _run_flowsom_benchmark

from ..utils._mouse_lineages_utils import _create_dataset

ORGAN = "bm"
INPUT_DIR = "../datasets/mouse_lineages/"
OUTPUT_DIR = f"../figure_data/Figure_3/mouse_lineages_{ORGAN}/flowsom_benchmark/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def run(continue_analysis: bool = False):
    dataset = _create_dataset(input_directory = INPUT_DIR,
                              organ = ORGAN)
    _run_flowsom_benchmark(dataset = dataset,
                           output_dir = OUTPUT_DIR,
                           gating_strategy = MOUSE_LINEAGE_GATING_STRATEGY,
                           gate_mapping = MOUSE_LINEAGE_GATE_MAPPING,
                           continue_analysis = continue_analysis)

if __name__ == "__main__":
    run()