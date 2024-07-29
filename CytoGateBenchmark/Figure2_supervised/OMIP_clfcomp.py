import os

from ..supervised_learning._classifier_comparison import _run_classifier_comparison
from ..utils._OMIP_utils import _create_dataset

INPUT_DIR = "../datasets/OMIP/"
OUTPUT_DIR = f"../figure_data/Figure_2/OMIP/classifier_comparison/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def run(continue_analysis: bool = False):
    dataset = _create_dataset(input_directory = INPUT_DIR)
    _run_classifier_comparison(dataset = dataset,
                               output_dir = OUTPUT_DIR,
                               continue_analysis = continue_analysis)

if __name__ == "__main__":
    run()


