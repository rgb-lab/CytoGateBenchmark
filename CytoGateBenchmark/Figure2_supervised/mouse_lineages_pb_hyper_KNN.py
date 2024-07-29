import os

from ..supervised_learning._hyperparameter_tuning import _run_hyperparameter_tuning
from ..utils._mouse_lineages_utils import _create_dataset

ORGAN = "pb"
INPUT_DIR = "../datasets/mouse_lineages/"
OUTPUT_DIR = f"../figure_data/Figure_2/mouse_lineages_{ORGAN}/hyper_KNN/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def run(continue_analysis: bool = False):
    dataset = _create_dataset(input_directory = INPUT_DIR,
                              organ = ORGAN)
    _run_hyperparameter_tuning(dataset = dataset,
                               output_dir = OUTPUT_DIR,
                               classifier = "KNN",
                               continue_analysis = continue_analysis)

if __name__ == "__main__":
    run()
