import os

from ..supervised_learning._classifier_comparison import _run_classifier_comparison
from ..utils._human_t_cells_utils import _create_dataset

INPUT_DIR = "../datasets/human_T_cells/"
OUTPUT_DIR = f"../figure_data/Figure_2/human_T_cells/classifier_comparison/"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def run(continue_analysis: bool = False):
    dataset = _create_dataset(input_directory = INPUT_DIR)
    _run_classifier_comparison(dataset = dataset,
                               output_dir = OUTPUT_DIR,
                               continue_analysis = continue_analysis)

if __name__ == "__main__":
    run()


