import os

from ._gating_strategies import ZPM_GATE_MAPPING, ZPM_GATING_STRATEGY
from ..unsupervised_learning._leiden_cluster_benchmark import _run_leiden_benchmark

from ..utils._ZPM_utils import _create_dataset

INPUT_DIR = "../datasets/ZPM/"
OUTPUT_DIR = f"../figure_data/Figure_3/ZPM/leiden_benchmark/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def run(continue_analysis: bool = False):
    dataset = _create_dataset(input_directory = INPUT_DIR)
    _run_leiden_benchmark(dataset = dataset,
                          output_dir = OUTPUT_DIR,
                          gating_strategy = ZPM_GATING_STRATEGY,
                          gate_mapping = ZPM_GATE_MAPPING,
                          continue_analysis = continue_analysis)

if __name__ == "__main__":
    run()