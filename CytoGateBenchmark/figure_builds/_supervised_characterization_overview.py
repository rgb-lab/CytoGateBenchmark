import os

import os
import numpy as np
import pandas as pd
from PIL import Image
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import seaborn as sns

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec


from ._utils import (_figure_label,
                     _remove_gate_comma_in_file,
                     _crop_whitespace,
                     FIGURE_HEIGHT_FULL,
                     FIGURE_HEIGHT_HALF,
                     FIGURE_WIDTH_FULL,
                     DATASETS_TO_USE,
                     DPI,
                     DATASETS_TO_USE,
                     DATASET_NAMING_MAP)
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from FACSPy.plotting._utils import (_map_obs_to_cmap,
                                    ANNOTATION_CMAPS,
                                    _scale_cbar_to_heatmap,
                                    _add_categorical_legend_to_clustermap)

DATASET_DIR = os.path.join(os.getcwd(), "figure_data/Figure_2")

YLABEL_FONTSIZE = 10
SCORELABEL_FONTSIZE = 10
LEGEND_FONTSIZE = 10

def _get_technical_data() -> pd.DataFrame:
    tmp = pd.DataFrame()
    for dataset in DATASETS_TO_USE:
        score_file = os.path.join(DATASET_DIR, dataset, "classifier_comparison/Technicals.log")
        _remove_gate_comma_in_file(score_file)
        data = pd.read_csv(score_file, index_col = False)
        data["mem_used"] = data["max_mem"] - data["min_mem"]
        data["total_time"] = data["train_time"] + data["pred_time_train"] + data["pred_time_test"] + data["pred_time_val"]
        data["dataset"] = DATASET_NAMING_MAP[dataset]
        tmp = pd.concat([tmp, data], axis = 0)
    return tmp

def _read_all_scores():
    first_frame = True
    for classifier in ["hyper_DT", "hyper_ET", "hyper_ETS", "hyper_RF", "hyper_MLP", "hyper_KNN"]:
        for dataset in DATASETS_TO_USE:
            data_path = f"{DATASET_DIR}/{dataset}/{classifier}/Scores.log"
            _remove_gate_comma_in_file(data_path)
            data = pd.read_csv(data_path, index_col = False)
            data["dataset"] = dataset
            if first_frame:
                full_data = data
                first_frame = False
            else:
                full_data = pd.concat([full_data, data], axis = 0)
    full_data = full_data[full_data["score_on"] == "val"]
    full_data = full_data[full_data["sampling"] == False]
    full_data["dataset"] = full_data["dataset"].map(DATASET_NAMING_MAP)
    full_data["tuned"] = full_data["tuned"].map({True: "tuned", False: "not tuned"})
    return full_data

def _generate_characterization_comparison(fig: Figure,
                                          ax: Axes,
                                          gs: GridSpec,
                                          subfigure_label: str):
    
    data = _read_all_scores()
    data = data[data["tuned"] == False]
    df = data.groupby(["dataset", "algorithm", "train_size", "tuned"]).median("f1_score").reset_index()
    df = df.pivot(index='dataset', columns=["algorithm", "train_size", "tuned"], values='f1_score')
    df = df.T.reset_index()

    metadata_annotation = ["algorithm", "train_size", "tuned"]
    metadata_annotation = ["algorithm", "train_size"]
    col_colors= [
        _map_obs_to_cmap(df, group, ANNOTATION_CMAPS[i])
                         for i, group in enumerate(metadata_annotation)
    ]

    clustermap = sns.clustermap(df[data["dataset"].unique()].T,
                                cmap = "Reds",
                                col_colors = col_colors,
                                row_cluster = False,
                                col_cluster = False,
                                figsize = (6, 4.2),
                                vmin = 0.3,
                                vmax = 1,
                                cbar_kws = {"label": "f1_score", "orientation": 'horizontal'},
                                colors_ratio=0.03,
                                yticklabels=True)
    cax = clustermap.ax_heatmap
    cax.yaxis.set_ticks_position("left")
    cax.xaxis.set_ticks_position("none")
    cax.set_xticklabels([])
    cax.set_yticklabels(cax.get_yticklabels(), fontsize = YLABEL_FONTSIZE)
    cax.set_ylabel("")
    _add_categorical_legend_to_clustermap(clustermap, cax, df, metadata_annotation)
    _scale_cbar_to_heatmap(clustermap, cax.get_position(), cbar_padding = 0.6, loc = "bottom")
    clustermap.ax_cbar.tick_params(labelsize=6)
    clustermap.ax_cbar.set_xlabel("F1 score", fontsize = SCORELABEL_FONTSIZE)
    clustermap.savefig('characterization_heatmap.png', dpi = 1200,
                       pad_inches = 0,
                       bbox_inches = "tight")
    plt.close()

    ax.axis("off")
    _figure_label(ax, subfigure_label, x = -0.1)
    fig_sgs = gs.subgridspec(1,1)

    img = _crop_whitespace(Image.open("characterization_heatmap.png"))
    axis = fig.add_subplot(fig_sgs[0])
    axis.imshow(img)
    axis.axis("off")
    axis = fig.add_subplot(axis)

    return
    
def _generate_ram_comparison(fig: Figure,
                              ax: Axes,
                              gs: GridSpec,
                              subfigure_label: str):
    data = _get_technical_data()
    df = data.groupby(["dataset", "algorithm"]).mean("f1_score").reset_index()
    df = df.pivot(index = "algorithm", columns = ["dataset"], values = "mem_used")
    df = df.T.reset_index()

    metadata_annotation = ["dataset"]
    col_colors= [
        _map_obs_to_cmap(df, group, ANNOTATION_CMAPS[i])
                         for i, group in enumerate(metadata_annotation)
    ]
    clustermap = sns.clustermap(df[data["algorithm"].unique()].T,
                                cmap = "viridis",
                                col_colors = col_colors,
                                row_cluster = False,
                                col_cluster = False,
                                figsize = (6, 4.2),
                                cbar_kws = {"label": "f1_score", "orientation": 'horizontal'},
                                colors_ratio=0.03,
                                yticklabels=True,
                                norm = SymLogNorm(linthresh = 1))
    cax = clustermap.ax_heatmap
    cax.yaxis.set_ticks_position("left")
    cax.xaxis.set_ticks_position("none")
    cax.set_xticklabels([])
    cax.set_yticklabels(cax.get_yticklabels(), fontsize = YLABEL_FONTSIZE)
    cax.set_ylabel("")
    _add_categorical_legend_to_clustermap(clustermap, cax, df, metadata_annotation)
    _scale_cbar_to_heatmap(clustermap, cax.get_position(), cbar_padding = 0.6, loc = "bottom")
    clustermap.ax_cbar.tick_params(labelsize=8)
    clustermap.ax_cbar.set_xlabel("Memory used [MB]", fontsize = SCORELABEL_FONTSIZE)
    clustermap.savefig('time_heatmap.png', dpi = 1200,
                       pad_inches = 0,
                       bbox_inches = "tight")
    plt.close()

    ax.axis("off")
    _figure_label(ax, subfigure_label, x = -0.1)
    fig_sgs = gs.subgridspec(1,1)

    img = _crop_whitespace(Image.open("time_heatmap.png"))
    axis = fig.add_subplot(fig_sgs[0])
    axis.imshow(img)
    axis.axis("off")
    axis = fig.add_subplot(axis)

    return

def _generate_time_comparison(fig: Figure,
                              ax: Axes,
                              gs: GridSpec,
                              subfigure_label: str):
    data = _get_technical_data()
    df = data.groupby(["dataset", "algorithm"]).mean("f1_score").reset_index()
    df = df.pivot(index = "algorithm", columns = ["dataset"], values = "total_time")
    df = df.T.reset_index()

    metadata_annotation = ["dataset"]
    col_colors= [
        _map_obs_to_cmap(df, group, ANNOTATION_CMAPS[i])
                         for i, group in enumerate(metadata_annotation)
    ]
    clustermap = sns.clustermap(df[data["algorithm"].unique()].T,
                                cmap = "viridis",
                                col_colors = col_colors,
                                row_cluster = False,
                                col_cluster = False,
                                figsize = (6, 4.2),
                                cbar_kws = {"label": "f1_score", "orientation": 'horizontal'},
                                colors_ratio=0.03,
                                yticklabels=True,
                                norm = LogNorm())
    cax = clustermap.ax_heatmap
    cax.yaxis.set_ticks_position("left")
    cax.xaxis.set_ticks_position("none")
    cax.set_xticklabels([])
    cax.set_yticklabels(cax.get_yticklabels(), fontsize = YLABEL_FONTSIZE)
    cax.set_ylabel("")
    _add_categorical_legend_to_clustermap(clustermap, cax, df, metadata_annotation)
    _scale_cbar_to_heatmap(clustermap, cax.get_position(), cbar_padding = 0.6, loc = "bottom")
    clustermap.ax_cbar.tick_params(labelsize=6)
    clustermap.ax_cbar.set_xlabel("Total time [s]", fontsize = SCORELABEL_FONTSIZE)
    clustermap.savefig('time_heatmap.png', dpi = 1200,
                       pad_inches = 0,
                       bbox_inches = "tight")
    plt.close()

    ax.axis("off")
    _figure_label(ax, subfigure_label, x = -0.1)
    fig_sgs = gs.subgridspec(1,1)

    img = _crop_whitespace(Image.open("time_heatmap.png"))
    axis = fig.add_subplot(fig_sgs[0])
    axis.imshow(img)
    axis.axis("off")
    axis = fig.add_subplot(axis)

    return


def _generate_f1_comparison(fig: Figure,
                            ax: Axes,
                            gs: GridSpec,
                            subfigure_label: str):
    tmp = pd.DataFrame()
    for dataset in DATASETS_TO_USE:
        score_file = os.path.join(DATASET_DIR, dataset, "classifier_comparison/Scores.log")
        _remove_gate_comma_in_file(score_file)
        data = pd.read_csv(score_file, index_col = False)
        data = data[data["score_on"] == "val"]
        data["dataset"] = DATASET_NAMING_MAP[dataset]
        tmp = pd.concat([tmp, data], axis = 0)
    data = tmp
    df = data.groupby(["dataset", "algorithm"]).mean("f1_score").reset_index()
    df = df.pivot(index = "algorithm", columns = ["dataset"], values = "f1_score")
    df = df.T.reset_index()

    metadata_annotation = ["dataset"]
    col_colors= [
        _map_obs_to_cmap(df, group, ANNOTATION_CMAPS[i])
                         for i, group in enumerate(metadata_annotation)
    ]
    clustermap = sns.clustermap(df[data["algorithm"].unique()].T,
                                cmap = "Reds",
                                col_colors = col_colors,
                                row_cluster = False,
                                col_cluster = False,
                                figsize = (6, 4.2),
                                vmin = 0,
                                vmax = 1,
                                cbar_kws = {"label": "f1_score", "orientation": 'horizontal'},
                                colors_ratio=0.03,
                                yticklabels=True)
    cax = clustermap.ax_heatmap
    cax.yaxis.set_ticks_position("left")
    cax.xaxis.set_ticks_position("none")
    cax.set_xticklabels([])
    cax.set_yticklabels(cax.get_yticklabels(), fontsize = YLABEL_FONTSIZE)
    cax.set_ylabel("")
    _add_categorical_legend_to_clustermap(clustermap, cax, df, metadata_annotation)
    _scale_cbar_to_heatmap(clustermap, cax.get_position(), cbar_padding = 0.6, loc = "bottom")
    clustermap.ax_cbar.tick_params(labelsize=6)
    clustermap.ax_cbar.set_xlabel("F1 score", fontsize = SCORELABEL_FONTSIZE)
    clustermap.savefig('f1_score_heatmap.png', dpi = 1200,
                       pad_inches = 0,
                       bbox_inches = "tight")
    plt.close()

    ax.axis("off")
    _figure_label(ax, subfigure_label, x = -0.1)
    fig_sgs = gs.subgridspec(1,1)

    img = _crop_whitespace(Image.open("f1_score_heatmap.png"))
    axis = fig.add_subplot(fig_sgs[0])
    axis.imshow(img)
    axis.axis("off")
    axis = fig.add_subplot(axis)

    return

def generate_classifier_overview_old(save: str = None,
                                     show: bool = True):

    fig = plt.figure(layout = "constrained",
                     figsize = (FIGURE_WIDTH_FULL, FIGURE_HEIGHT_FULL))
    gs = GridSpec(ncols = 1,
                  nrows = 3,
                  figure = fig,
                  height_ratios = [1, 1, 1])
    
    a_coords = gs[0, 0]
    b_coords = gs[1, 0]
    c_coords = gs[2, 0]

    fig_a = fig.add_subplot(a_coords)
    fig_b = fig.add_subplot(b_coords)
    fig_c = fig.add_subplot(c_coords)

    _generate_f1_comparison(fig = fig,
                            ax = fig_a,
                            gs = a_coords,
                            subfigure_label = "A")
    _generate_time_comparison(fig = fig,
                              ax = fig_b,
                              gs = b_coords,
                              subfigure_label = "B")
    _generate_ram_comparison(fig = fig,
                             ax = fig_c,
                             gs = c_coords,
                             subfigure_label = "C")

    if save:
        plt.savefig(os.path.join(os.getcwd(), save),
                    dpi = DPI,
                    bbox_inches = "tight")
        
    if show:
        plt.show()

    return

def generate_classifier_overview(save: str = None,
                                 show: bool = True):

    fig = plt.figure(layout = "constrained",
                     figsize = (FIGURE_WIDTH_FULL, FIGURE_HEIGHT_HALF))
    gs = GridSpec(ncols = 2,
                  nrows = 2,
                  figure = fig,
                  height_ratios = [1, 1])
    
    a_coords = gs[0, 0]
    b_coords = gs[0, 1]
    c_coords = gs[1, 0]
    d_coords = gs[1, 1]

    fig_a = fig.add_subplot(a_coords)
    fig_b = fig.add_subplot(b_coords)
    fig_c = fig.add_subplot(c_coords)
    fig_d = fig.add_subplot(d_coords)

    _generate_f1_comparison(fig = fig,
                            ax = fig_a,
                            gs = a_coords,
                            subfigure_label = "A")
    _generate_time_comparison(fig = fig,
                              ax = fig_b,
                              gs = b_coords,
                              subfigure_label = "B")
    _generate_ram_comparison(fig = fig,
                             ax = fig_c,
                             gs = c_coords,
                             subfigure_label = "C")
    _generate_characterization_comparison(fig = fig,
                                          ax = fig_d,
                                          gs = d_coords,
                                          subfigure_label = "D")

    if save:
        plt.savefig(os.path.join(os.getcwd(), save),
                    dpi = DPI,
                    bbox_inches = "tight")
        
    if show:
        plt.show()

    return
