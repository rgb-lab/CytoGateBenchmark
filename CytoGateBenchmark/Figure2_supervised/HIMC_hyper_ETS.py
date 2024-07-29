import os

from ..supervised_learning._hyperparameter_tuning import _run_hyperparameter_tuning
from ..utils._HIMC_utils import _create_dataset

INPUT_DIR = "../datasets/HIMC/"
OUTPUT_DIR = f"../figure_data/Figure_2/HIMC/hyper_ETS/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def run(continue_analysis: bool = False):
    dataset = _create_dataset(input_directory = INPUT_DIR)
    _run_hyperparameter_tuning(dataset = dataset,
                               output_dir = OUTPUT_DIR,
                               classifier = "ExtraTreesClassifier",
                               continue_analysis = continue_analysis)

if __name__ == "__main__":
    run()
