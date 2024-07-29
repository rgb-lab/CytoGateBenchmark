import os

import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from typing import Literal, Optional

from ._utils import (_figure_label,
                     _remove_gate_comma_in_file,
                     _extract_population_from_gates,
                     STRIPPLOT_PARAMS,
                     BOXPLOT_PARAMS,
                     XTICKLABEL_PARAMS,
                     TICKPARAMS_PARAMS,
                     CENTERED_LEGEND_PARAMS,
                     FIGURE_HEIGHT_FULL,
                     FIGURE_WIDTH_FULL,
                     TITLE_SIZE,
                     AXIS_LABEL_SIZE,
                     UNSUPERVISED_SCORE,
                     SCORING_YLIMS,
                     DPI)

RAW_INPUT_DIR = os.path.join(os.getcwd(), "datasets/")
DATASET_DIR = os.path.join(os.getcwd(), "figure_data/Figure_3")

SCORE_LABEL = UNSUPERVISED_SCORE.replace("_", " ")

DATASET_MAP = {
    "A": "leiden",
    "B": "PARC",
    "C": "PhenoGraph",
    "D": "FlowSOM"
}

STRIPPLOT_PARAMS["s"] = 2

def _read_technical_data(dataset_name: str,
                         assay: str) -> pd.DataFrame:
    file_path = os.path.join(DATASET_DIR, dataset_name, assay, "Technicals.log")
    _remove_gate_comma_in_file(file_path)
    data = pd.read_csv(file_path, index_col = False)
    data["mem_used"] = data["max_mem"] - data["min_mem"]
    return data

def _read_score_data(dataset_name: str,
                     assay: str,
                     gates_to_use: list[str]) -> pd.DataFrame:
    file_path = os.path.join(DATASET_DIR, dataset_name, assay, "Scores.log")
    _remove_gate_comma_in_file(file_path)
    data = pd.read_csv(file_path, index_col = False)
    data = data[data["gate"].isin(gates_to_use)]
    if any("/" in gate for gate in data["gate"]):
        data["gate"] = _extract_population_from_gates(data["gate"].tolist())
    return data

def _generate_comparison_figure(fig: Figure,
                                ax: Axes,
                                gs: GridSpec,
                                subfigure_label: str,
                                score_data: pd.DataFrame,
                                technical_data: pd.DataFrame,
                                x_param):
    assay = DATASET_MAP[subfigure_label]
    ax.axis("off")
    _figure_label(ax, subfigure_label, x = -0.2)
    fig_sgs = gs.subgridspec(1,12)

    axis = fig.add_subplot(fig_sgs[0,0:6])
    plot_kwargs = {
        "data": score_data,
        "x": x_param,
        "y": UNSUPERVISED_SCORE,
        "hue": "gate",
        "ax": axis
    }

    sns.stripplot(**plot_kwargs,
                  **STRIPPLOT_PARAMS)
    sns.boxplot(**plot_kwargs,
                **BOXPLOT_PARAMS)
    
    axis.set_title(f"Classification Accuracy:\n{assay}", fontsize = TITLE_SIZE)
    axis.set_xlabel(axis.get_xlabel(), fontsize = AXIS_LABEL_SIZE)
    axis.set_ylabel(SCORE_LABEL, fontsize = AXIS_LABEL_SIZE)
    axis.set_ylim(SCORING_YLIMS)
    handles, _ = axis.get_legend_handles_labels()
    axis.legend(handles, score_data["gate"].unique(),
                **CENTERED_LEGEND_PARAMS)
    axis.tick_params(**TICKPARAMS_PARAMS)
    axis.set_xticklabels(axis.get_xticklabels(),
                         **XTICKLABEL_PARAMS)

    axis = fig.add_subplot(fig_sgs[0,6:9])
    plot_kwargs = {
        "data": technical_data,
        "x": x_param,
        "y": "train_time",
        "ax": axis
    }

    sns.stripplot(**plot_kwargs,
                  **STRIPPLOT_PARAMS)
    sns.boxplot(**plot_kwargs,
                **BOXPLOT_PARAMS)
    
    axis.set_title(f"Classification Time:\n{assay}", fontsize = TITLE_SIZE)
    axis.set_xlabel(axis.get_xlabel(), fontsize = AXIS_LABEL_SIZE)
    axis.set_ylabel("train time [s]", fontsize = AXIS_LABEL_SIZE)
    axis.tick_params(**TICKPARAMS_PARAMS)
    axis.set_xticklabels(axis.get_xticklabels(),
                         **XTICKLABEL_PARAMS)

    axis = fig.add_subplot(fig_sgs[0,9:12])
    plot_kwargs = {
        "data": technical_data,
        "x": x_param,
        "y": "mem_used",
        "ax": axis
    }

    sns.stripplot(**plot_kwargs,
                  **STRIPPLOT_PARAMS)
    sns.boxplot(**plot_kwargs,
                **BOXPLOT_PARAMS)
    axis.set_title(f"Memory Consumption:\n{assay}", fontsize = TITLE_SIZE)
    axis.set_xlabel(axis.get_xlabel(), fontsize = AXIS_LABEL_SIZE)
    axis.set_ylabel("memory used [MB]", fontsize = AXIS_LABEL_SIZE)
    axis.tick_params(**TICKPARAMS_PARAMS)
    axis.set_xticklabels(axis.get_xticklabels(),
                         **XTICKLABEL_PARAMS)

def _generate_figure_d(fig: Figure,
                       ax: Axes,
                       gs: GridSpec,
                       subfigure_label: str,
                       score_data: pd.DataFrame,
                       technical_data: pd.DataFrame):
    _generate_comparison_figure(fig = fig,
                                ax = ax,
                                gs = gs,
                                subfigure_label = subfigure_label,
                                score_data = score_data,
                                technical_data = technical_data,
                                x_param = "n_clusters")

def _generate_figure_c(fig: Figure,
                       ax: Axes,
                       gs: GridSpec,
                       subfigure_label: str,
                       score_data: pd.DataFrame,
                       technical_data: pd.DataFrame):
    _generate_comparison_figure(fig = fig,
                                ax = ax,
                                gs = gs,
                                subfigure_label = subfigure_label,
                                score_data = score_data,
                                technical_data = technical_data,
                                x_param = "resolution")

def _generate_figure_b(fig: Figure,
                       ax: Axes,
                       gs: GridSpec,
                       subfigure_label: str,
                       score_data: pd.DataFrame,
                       technical_data: pd.DataFrame):
    _generate_comparison_figure(fig = fig,
                                ax = ax,
                                gs = gs,
                                subfigure_label = subfigure_label,
                                score_data = score_data,
                                technical_data = technical_data,
                                x_param = "resolution")


def _generate_figure_a(fig: Figure,
                       ax: Axes,
                       gs: GridSpec,
                       subfigure_label: str,
                       score_data: pd.DataFrame,
                       technical_data: pd.DataFrame):
    _generate_comparison_figure(fig = fig,
                                ax = ax,
                                gs = gs,
                                subfigure_label = subfigure_label,
                                score_data = score_data,
                                technical_data = technical_data,
                                x_param = "resolution")
    
    return
    

def generate_classifier_characterization(dataset_name: Literal[
                                            "mouse_lineages_bm",
                                            "mouse_lineages_spl",
                                            "mouse_lineages_pb",
                                            "HIMC",
                                            "ZPM",
                                            "human_t_cells",
                                            "OMIP"
                                            ],
                                         gates_to_use: list[str],
                                         save: Optional[str] = None,
                                         show: bool = True):
    
    if dataset_name != "human_t_cells":
        gates_to_use = _extract_population_from_gates(gates_to_use)
    
    leiden_benchmark = _read_score_data(dataset_name = dataset_name,
                                        assay = "leiden_benchmark",
                                        gates_to_use = gates_to_use)
    parc_benchmark = _read_score_data(dataset_name = dataset_name,
                                      assay = "parc_benchmark",
                                      gates_to_use = gates_to_use)
    phenograph_benchmark = _read_score_data(dataset_name = dataset_name,
                                            assay = "phenograph_benchmark",
                                            gates_to_use = gates_to_use)
    flowsom_benchmark = _read_score_data(dataset_name = dataset_name,
                                         assay = "flowsom_benchmark",
                                         gates_to_use = gates_to_use)

    leiden_technicals = _read_technical_data(dataset_name = dataset_name,
                                             assay = "leiden_benchmark")
    parc_technicals = _read_technical_data(dataset_name = dataset_name,
                                           assay = "parc_benchmark")
    phenograph_technicals = _read_technical_data(dataset_name = dataset_name,
                                                 assay = "phenograph_benchmark")
    flowsom_technicals = _read_technical_data(dataset_name = dataset_name,
                                              assay = "flowsom_benchmark")

    fig = plt.figure(layout = "constrained",
                     figsize = (FIGURE_WIDTH_FULL, FIGURE_HEIGHT_FULL))
    gs = GridSpec(ncols = 1,
                  nrows = 4,
                  figure = fig,
                  height_ratios = [1,1,1,1])
    
    a_coords = gs[0, :]
    b_coords = gs[1, :]
    c_coords = gs[2, :]
    d_coords = gs[3, :]

    fig_a = fig.add_subplot(a_coords)
    fig_b = fig.add_subplot(b_coords)
    fig_c = fig.add_subplot(c_coords)
    fig_d = fig.add_subplot(d_coords)

    _generate_figure_a(fig = fig,
                       ax = fig_a,
                       gs = a_coords,
                       subfigure_label = "A",
                       score_data = leiden_benchmark,
                       technical_data = leiden_technicals)
    _generate_figure_b(fig = fig,
                       ax = fig_b,
                       gs = b_coords,
                       subfigure_label = "B",
                       score_data = parc_benchmark,
                       technical_data = parc_technicals)
    _generate_figure_c(fig = fig,
                       ax = fig_c,
                       gs = c_coords,
                       subfigure_label = "C",
                       score_data = phenograph_benchmark,
                       technical_data = phenograph_technicals)
    _generate_figure_d(fig = fig,
                       ax = fig_d,
                       gs = d_coords,
                       subfigure_label = "D",
                       score_data = flowsom_benchmark,
                       technical_data = flowsom_technicals)

    if save:
        plt.savefig(os.path.join(os.getcwd(), save),
                    dpi = DPI,
                    bbox_inches = "tight")
        
    if show:
        plt.show()

    return