import os

from ._gating_strategies import OMIP_GATE_MAPPING, OMIP_GATING_STRATEGY
from ..unsupervised_learning._flowsom_cluster_benchmark import _run_flowsom_benchmark

from ..utils._OMIP_utils import _create_dataset

INPUT_DIR = "../datasets/OMIP/"
OUTPUT_DIR = f"../figure_data/Figure_3/OMIP/flowsom_benchmark/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def run(continue_analysis: bool = False):
    dataset = _create_dataset(input_directory = INPUT_DIR)
    _run_flowsom_benchmark(dataset = dataset,
                           output_dir = OUTPUT_DIR,
                           gating_strategy = OMIP_GATING_STRATEGY,
                           gate_mapping = OMIP_GATE_MAPPING,
                           continue_analysis = continue_analysis)

if __name__ == "__main__":
    run()