import os

from ._gating_strategies import HIMC_GATING_STRATEGY, HIMC_GATE_MAPPING
from ..unsupervised_learning._parc_cluster_benchmark import _run_parc_benchmark

from ..utils._HIMC_utils import _create_dataset

INPUT_DIR = "../datasets/HIMC/"
OUTPUT_DIR = f"../figure_data/Figure_3/HIMC/parc_benchmark/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def run(continue_analysis: bool = False):
    dataset = _create_dataset(input_directory = INPUT_DIR)
    _run_parc_benchmark(dataset = dataset,
                        output_dir = OUTPUT_DIR,
                        gating_strategy = HIMC_GATING_STRATEGY,
                        gate_mapping = HIMC_GATE_MAPPING,
                        full_gate_provided = False,
                        continue_analysis = continue_analysis)

if __name__ == "__main__":
    run()