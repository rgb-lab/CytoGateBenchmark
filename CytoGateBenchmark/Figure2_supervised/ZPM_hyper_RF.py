import os

from ..supervised_learning._hyperparameter_tuning import _run_hyperparameter_tuning
from ..utils._ZPM_utils import _create_dataset

INPUT_DIR = "../datasets/ZPM/"
OUTPUT_DIR = f"../figure_data/Figure_2/ZPM/hyper_RF/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def run(continue_analysis: bool = False):
    dataset = _create_dataset(input_directory = INPUT_DIR)
    _run_hyperparameter_tuning(dataset = dataset,
                               output_dir = OUTPUT_DIR,
                               classifier = "RandomForestClassifier",
                               continue_analysis = continue_analysis)

if __name__ == "__main__":
    run()

