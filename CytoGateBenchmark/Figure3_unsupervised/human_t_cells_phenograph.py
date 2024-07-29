import os

from ._gating_strategies import GIESE_TA_GATING_STRATEGY, GIESE_TA_GATE_MAP
from ..unsupervised_learning._phenograph_cluster_benchmark import _run_phenograph_benchmark

from ..utils._human_t_cells_utils import _create_dataset

INPUT_DIR = "../datasets/human_T_cells/"
OUTPUT_DIR = f"../figure_data/Figure_3/human_T_cells/phenograph_benchmark/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def run(continue_analysis: bool = False):
    dataset = _create_dataset(input_directory = INPUT_DIR)
    _run_phenograph_benchmark(dataset = dataset,
                              output_dir = OUTPUT_DIR,
                              gating_strategy = GIESE_TA_GATING_STRATEGY,
                              gate_mapping = GIESE_TA_GATE_MAP,
                              full_gate_provided = True,
                              continue_analysis = continue_analysis)

if __name__ == "__main__":
    run()