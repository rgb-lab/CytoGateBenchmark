import os
import numpy as np
import pandas as pd
from anndata import AnnData
import FACSPy as fp
import scanpy as sc
from matplotlib.axes import Axes
from matplotlib.text import Text

from typing import Literal, Union

from PIL import Image, ImageChops

AXIS_LABEL_SIZE = 6
TITLE_SIZE = 8
UMAP_LABEL_SIZE = 6

DATASETS_TO_USE = [
    "mouse_lineages_bm",
    "mouse_lineages_pb",
    "mouse_lineages_spl",
    "human_t_cells",
    "ZPM",
    "OMIP",
    "HIMC"
]

DATASET_NAMING_MAP = {
    "mouse_lineages_bm": "Mouse Lineages BM",
    "mouse_lineages_pb": "Mouse Lineages PB",
    "mouse_lineages_spl": "Mouse Lineages SPL",
    "human_t_cells": "Human T-Cells",
    "ZPM": "ZPM",
    "OMIP": "OMIP",
    "HIMC": "HIMC"
}

DPI = 300

FIGURE_WIDTH_FULL = 6.75
FIGURE_WIDTH_HALF = FIGURE_WIDTH_FULL / 2

FIGURE_HEIGHT_FULL = 9.375
FIGURE_HEIGHT_HALF = FIGURE_HEIGHT_FULL / 2

SUPERVISED_SCORE = "f1_score"
UNSUPERVISED_SCORE = "jaccard_score"

#SCORING_YLIMS = (-0.25 , 1.25)
SCORING_YLIMS = (-0.15, 1.05)

TRAIN_SIZES = ["50", "500", "5000", "50000", "500000"]

SUPERVISED_UMAP_PALETTE = "Set1"

STRIPPLOT_PARAMS = {
    "linewidth": 0.5,
    "dodge": True,
    "s": 2
}

BOXPLOT_PARAMS = {
    "boxprops": dict(facecolor = "white"),
    "whis": (10,90),
    "linewidth": 1,
    "showfliers": False
}

XTICKLABEL_PARAMS = {
    "ha": "right",
    "rotation": 45,
    "rotation_mode": "anchor",
    "fontsize": AXIS_LABEL_SIZE
}

TICKPARAMS_PARAMS = {
    "axis": "both",
    "labelsize": AXIS_LABEL_SIZE
}

CENTERED_LEGEND_PARAMS = {
    "bbox_to_anchor": (1, 0.5),
    "loc": "center left",
    "fontsize": AXIS_LABEL_SIZE,
    "markerscale": 0.5
}

def _crop_whitespace(im):
    pix = np.asarray(im)
    
    pix = pix[:,:,0:3] # Drop the alpha channel
    idx = np.where(pix-255)[0:2] # Drop the color when finding edges
    box = list(map(min,idx))[::-1] + list(map(max,idx))[::-1]
    
    region = im.crop(box)
    region_pix = np.asarray(region)
    return region_pix


def _scanpy_vector_friendly():
    sc.set_figure_params(vector_friendly = True, dpi = DPI)
    return

def _get_gating_information(dataset_name: str) -> tuple[dict[str: dict], dict[str: dict]]:
    from ..Figure3_unsupervised import _gating_strategies as gs
    if "mouse_lineages" in dataset_name:
        return gs.MOUSE_LINEAGE_GATING_STRATEGY, gs.MOUSE_LINEAGE_GATE_MAPPING
    elif dataset_name == "OMIP":
        return gs.OMIP_GATING_STRATEGY, gs.OMIP_GATE_MAPPING
    elif dataset_name == "HIMC":
        return gs.HIMC_GATING_STRATEGY, gs.HIMC_GATE_MAPPING
    elif dataset_name == "human_t_cells":
        return gs.GIESE_TA_GATING_STRATEGY, gs.GIESE_TA_GATE_MAP
    elif dataset_name == "ZPM":
        return gs.ZPM_GATING_STRATEGY, gs.ZPM_GATE_MAPPING
    return

def _extract_population_from_gates_ticklabels(gates: list[Text]) -> None:
    for gate in gates:
        gate._text = _extract_population_from_gates(gate._text)
    return

def _remove_underscores_from_gates_ticklabels(gates: list[Text]) -> list[Text]:
    for gate in gates:
        gate._text = _remove_underscores_from_gates(gate._text)
    return gates

def _extract_population_from_gates(gates: Union[list[str], str]) -> Union[list[str], str]:
    if isinstance(gates, list) or isinstance(gates, pd.Index) or isinstance(gates, pd.Series) or isinstance(gates, np.ndarray):
        return [gate.split("/")[-1] for gate in gates]
    if isinstance(gates, str):
        return gates.split("/")[-1]
    print(type(gates))
    raise ValueError("Something went wrong")

def _remove_underscores_from_gates(gates: Union[list[str], str]) -> Union[list[str], str]:
    if isinstance(gates, list) or isinstance(gates, pd.Index) or isinstance(gates, pd.Series) or isinstance(gates, np.ndarray):
        return [gate.replace("_", " ") for gate in gates]
    if isinstance(gates, str):
        return gates.replace("_", " ")
    print(type(gates))
    raise ValueError("Something went wrong")
    
def _figure_label(ax: Axes, label, x = 0, y = 1):
    """labels individual subfigures. Requires subgrid to not use figure axis coordinates."""
    ax.text(x,y,label, fontsize = 12)
    return

def _trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2, 0)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    return im

def _remove_gate_comma_in_file(path):
    new_path = os.path.join(os.path.dirname(path), "Scores_.log")
    with open(path, "r") as in_file:
        with open(os.path.join(new_path), "w") as out_file:
            for line in in_file:
                out_file.write(line.replace("_,_", "_"))
    os.remove(path)
    os.rename(new_path, path)
    return

def _dataset_exists(input_directory: str):
    return os.path.isfile(os.path.join(input_directory, "raw_data.h5ad"))

def _generate_dataset(dataset_name: Literal[
                      "mouse_lineages_bm",
                      "mouse_lineages_spl",
                      "mouse_lineages_pb",
                      "HIMC",
                      "ZPM",
                      "human_t_cells",
                      "OMIP"
                      ],
                      DATASET_DIR: str,
                      RAW_INPUT_DIR: str) -> AnnData:
    dataset_dir = os.path.join(DATASET_DIR, f"{dataset_name}")
    if "mouse_lineages" in dataset_name:
        raw_input_dir = os.path.join(RAW_INPUT_DIR, "mouse_lineages")
    else:
        raw_input_dir = os.path.join(RAW_INPUT_DIR, f"{dataset_name}")

    if _dataset_exists(dataset_dir):
        print("Loading dataset...")
        return fp.read_dataset(dataset_dir, "raw_data")
    if dataset_name == "mouse_lineages_bm":
        from ..utils._mouse_lineages_utils import _create_dataset
        dataset = _create_dataset(raw_input_dir, 
                                  organ = "bm")
        fp.save_dataset(dataset, dataset_dir, "raw_data")
        return dataset
    elif dataset_name == "mouse_lineages_pb":
        from ..utils._mouse_lineages_utils import _create_dataset
        dataset = _create_dataset(raw_input_dir, 
                                  organ = "pb")
        fp.save_dataset(dataset, dataset_dir, "raw_data")
        return dataset
    elif dataset_name == "mouse_lineages_spl":
        from ..utils._mouse_lineages_utils import _create_dataset
        dataset = _create_dataset(raw_input_dir, 
                                  organ = "spl")
        fp.save_dataset(dataset, dataset_dir, "raw_data")
        return dataset
    elif dataset_name == "HIMC":
        from ..utils._HIMC_utils import _create_dataset
        dataset = _create_dataset(raw_input_dir)
        fp.save_dataset(dataset, dataset_dir, "raw_data")
        return dataset
    elif dataset_name == "ZPM":
        from ..utils._ZPM_utils import _create_dataset
        dataset = _create_dataset(raw_input_dir)
        fp.save_dataset(dataset, dataset_dir, "raw_data")
        return dataset
    elif dataset_name == "human_t_cells":
        from ..utils._human_t_cells_utils import _create_dataset
        dataset = _create_dataset(raw_input_dir)
        fp.save_dataset(dataset, dataset_dir, "raw_data")
        return dataset
    elif dataset_name == "OMIP":
        from ..utils._OMIP_utils import _create_dataset
        dataset = _create_dataset(raw_input_dir)
        fp.save_dataset(dataset, dataset_dir, "raw_data")
        return dataset
    else:
        raise TypeError("dataset not recognized")
