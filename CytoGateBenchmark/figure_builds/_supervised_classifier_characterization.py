import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from PIL import Image

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from FACSPy.plotting._utils import (_map_obs_to_cmap,
                                    _has_interval_index,
                                    ANNOTATION_CMAPS,
                                    CONTINUOUS_CMAPS,
                                    _scale_cbar_to_heatmap)

from typing import Literal

from ._utils import (_figure_label,
                     _remove_gate_comma_in_file,
                     _crop_whitespace,
                     FIGURE_HEIGHT_FULL,
                     FIGURE_WIDTH_FULL,
                     STRIPPLOT_PARAMS,
                     BOXPLOT_PARAMS,
                     XTICKLABEL_PARAMS,
                     TICKPARAMS_PARAMS,
                     TITLE_SIZE,
                     AXIS_LABEL_SIZE,
                     SUPERVISED_SCORE,
                     TRAIN_SIZES,
                     SCORING_YLIMS,
                     DPI)

YLABEL_FONTSIZE = 8
SCORELABEL_FONTSIZE = 8
LEGEND_FONTSIZE = 10

HYPERPARAMETER_PATH_MAP = {
    "RandomForestClassifier": "hyper_RF",
    "DecisionTreeClassifier": "hyper_DT",
    "MLPClassifier": "hyper_MLP",
    "KNN": "hyper_KNN",
    "ExtraTreeClassifier": "hyper_ET",
    "ExtraTreesClassifier": "hyper_ETS"
}

SCORE_LABEL = SUPERVISED_SCORE.replace("_", " ")

RAW_INPUT_DIR = os.path.join(os.getcwd(), "datasets/")
DATASET_DIR = os.path.join(os.getcwd(), "figure_data/Figure_2")

from matplotlib.axes import Axes
from matplotlib.patches import Patch

def _add_categorical_legend_to_clustermap(clustermap: sns.matrix.ClusterGrid,
                                          heatmap: Axes,
                                          data: pd.DataFrame,
                                          annotate: list[str]) -> None:
    next_legend = 0
    for i, group in enumerate(annotate):
        group_lut = _map_obs_to_cmap(data,
                                     group,
                                     CONTINUOUS_CMAPS[i] if _has_interval_index(data[group])
                                                         else ANNOTATION_CMAPS[i],
                                     return_mapping = True)
        if _has_interval_index(data[group]):
            sorted_index = list(data[group].cat.categories.values)
            if np.nan in group_lut.keys():
                sorted_index = [np.nan] + sorted_index
            group_lut = {key: group_lut[key] for key in sorted_index}
        handles = [Patch(facecolor = group_lut[name]) for name in group_lut]
        legend_space = 0.062 * (len(handles) + 1)
        group_legend = heatmap.legend(handles,
                                      group_lut,
                                      title = group,
                                      loc = "upper left",
                                      fontsize = 6,
                                      title_fontsize = 8,
                                      bbox_to_anchor = (1.02, 1 - next_legend, 0, 0),
                                      bbox_transform = heatmap.transAxes
                                      )

        next_legend += legend_space
        clustermap.fig.add_artist(group_legend)
    return 

def _generate_hyperparameter_tuning_plot(fig: Figure,
                                         ax: Axes,
                                         gs: GridSpec,
                                         subfigure_label: str,
                                         dataset_name: str,
                                         classifier: str) -> None:
    ax.axis("off")
    _figure_label(ax, subfigure_label, x = -0.1)
    fig_sgs = gs.subgridspec(1,1)
    
    score = SUPERVISED_SCORE
    
    axis = fig.add_subplot(fig_sgs[0])
    
    score_file = os.path.join(DATASET_DIR, f"{dataset_name}/{HYPERPARAMETER_PATH_MAP[classifier]}/Scores.log")
    _remove_gate_comma_in_file(score_file)
    score_data = pd.read_csv(score_file,
                             index_col = False)
    score_data = score_data[score_data["score_on"] == "val"]
    score_data: pd.DataFrame = score_data.groupby(["sample_ID", "tuned"]).mean(SUPERVISED_SCORE)
    score_data = score_data.reset_index()
    score_data["tuned"] = score_data["tuned"].map({False: "not tuned",
                                                   True: "tuned"})
    plot_kwargs = {
        "data": score_data,
        "x": "tuned",
        "y": score,
        "ax": axis
    }
    sns.stripplot(**plot_kwargs,
                  **STRIPPLOT_PARAMS)
    sns.boxplot(**plot_kwargs,
                **BOXPLOT_PARAMS)
    ticklabels = axis.get_xticklabels()
    for label in ticklabels:
        label._text = label._text.split("/")[-1].replace("_", " ")
    axis.set_xticklabels(ticklabels,
                         **XTICKLABEL_PARAMS)
    axis.tick_params(**TICKPARAMS_PARAMS)

    classifier_prefix = classifier.split("Classifier")[0] if classifier != "KNN" else "KNN"
    axis.set_title(f"{classifier_prefix}:\nhyperparameter\ntuning",
                   fontsize = TITLE_SIZE)
    axis.set_xlabel("")
    axis.set_ylabel(f"mean {SCORE_LABEL}\nof all gates",
                    fontsize = AXIS_LABEL_SIZE)

    axis.legend().remove()
    axis = fig.add_subplot(axis)  

    return

def _read_score_file(dataset_name: str,
                     classifier: str,
                     gates_to_use: list[str]):

    hyperparameter_path = HYPERPARAMETER_PATH_MAP[classifier]
    score_file = os.path.join(DATASET_DIR, f"{dataset_name}/{hyperparameter_path}/Scores.log")

    _remove_gate_comma_in_file(score_file)
    score_data = pd.read_csv(score_file,
                             index_col = False)
    score_data = score_data[score_data["score_on"] == "val"]
    score_data = score_data[score_data["tuned"] == True]
    score_data = score_data[score_data["algorithm"] == classifier]
    score_data["train_size"] = score_data["train_size"].astype(str)
    score_data = score_data[score_data["train_size"].isin(TRAIN_SIZES)]
    score_data = score_data[score_data["gate"].isin(gates_to_use)]
    score_data = score_data.sort_values("train_size", ascending = True)

    return score_data


def _generate_train_size_comparison(fig: Figure,
                                    ax: Axes,
                                    gs: GridSpec,
                                    subfigure_label: str,
                                    dataset_name: str,
                                    classifier: str,
                                    gates_to_use: list[str]) -> None:
    ax.axis("off")
    _figure_label(ax, subfigure_label, x = -0.1)
    fig_sgs = gs.subgridspec(1,1)
    
    axis = fig.add_subplot(fig_sgs[0])

    score_data = _read_score_file(dataset_name,
                                  classifier,
                                  gates_to_use)
    
    plot_kwargs = {
        "data": score_data,
        "x": "gate",
        "y": SUPERVISED_SCORE,
        "hue": "train_size",
        "ax": axis
    }
    sns.stripplot(**plot_kwargs,
                  **STRIPPLOT_PARAMS)
    sns.boxplot(**plot_kwargs,
                **BOXPLOT_PARAMS)
    ticklabels = axis.get_xticklabels()
    for label in ticklabels:
        label._text = label._text.split("/")[-1].replace("_", " ")
    axis.set_xticklabels(ticklabels,
                         **XTICKLABEL_PARAMS)
    axis.tick_params(**TICKPARAMS_PARAMS)

    classifier_prefix = classifier.split("Classifier")[0] if classifier != "KNN" else "KNN"
    axis.set_title(f"{classifier_prefix}:\n{SCORE_LABEL} per gate", fontsize = TITLE_SIZE)
    axis.set_xlabel("")
    axis.set_ylabel(SCORE_LABEL, fontsize = AXIS_LABEL_SIZE)
    axis.set_ylim(SCORING_YLIMS)
    
    handles, _ = axis.get_legend_handles_labels()
    axis.legend(handles,
                TRAIN_SIZES,
                loc = "lower left",
                bbox_to_anchor = (0,0),
                fontsize = AXIS_LABEL_SIZE,
                title = "train set\nsize [n]",
                title_fontsize = AXIS_LABEL_SIZE,
                markerscale = 0.5)
    axis = fig.add_subplot(
        axis
    )
    return

def _generate_time_comparison(fig: Figure,
                              ax: Axes,
                              gs: GridSpec,
                              subfigure_label: str,
                              dataset_name: str) -> None:
    ax.axis("off")
    _figure_label(ax, subfigure_label, x = -0.1)
    fig_sgs = gs.subgridspec(1,1)
    
    axis = fig.add_subplot(fig_sgs[0])
    tech_data = pd.read_csv(os.path.join(DATASET_DIR,
                                         f"{dataset_name}/classifier_comparison/Technicals.log"),
                            index_col = False)
    tech_data["mem_used"] = tech_data["max_mem"] - tech_data["min_mem"]
    plot_kwargs = {
        "data": tech_data,
        "x": "algorithm",
        "y": "train_time",
        "ax": axis
    }
    sns.stripplot(**plot_kwargs,
                  **STRIPPLOT_PARAMS)
    sns.boxplot(**plot_kwargs,
                **BOXPLOT_PARAMS)
    ticklabels = axis.get_xticklabels()
    for label in ticklabels:
        label._text = label._text.split("/")[-1].replace("_", " ")
    axis.set_xticklabels(ticklabels,
                         **XTICKLABEL_PARAMS)
    axis.tick_params(**TICKPARAMS_PARAMS)
    axis.set_xlabel("")
    axis.set_ylabel(f"Train Time [s]", fontsize = AXIS_LABEL_SIZE)
    axis.set_title(f"Train Time per classifier", fontsize = TITLE_SIZE)
    axis.set_yscale("log")
    axis = fig.add_subplot(axis)
    return

def _generate_classifier_overview(fig: Figure,
                                  ax: Axes,
                                  gs: GridSpec,
                                  subfigure_label: str,
                                  dataset_name: str,
                                  gates_to_use: list[str]) -> None:
    ax.axis("off")
    _figure_label(ax, subfigure_label, x = -0.1)
    fig_sgs = gs.subgridspec(1,1)
    
    
    axis = fig.add_subplot(fig_sgs[0])
    score_file = os.path.join(DATASET_DIR,
                              f"{dataset_name}/classifier_comparison/Scores.log")
    _remove_gate_comma_in_file(score_file)
    score_data = pd.read_csv(score_file,
                             index_col = False)
    score_data = score_data[score_data["score_on"] == "val"]
    score_data = score_data[score_data["gate"].isin(gates_to_use)]
    plot_kwargs = {
        "data": score_data,
        "x": "algorithm",
        "y": SUPERVISED_SCORE,
        "ax": axis
    }
    sns.stripplot(**plot_kwargs,
                  **STRIPPLOT_PARAMS)
    sns.boxplot(**plot_kwargs,
                **BOXPLOT_PARAMS)
    ticklabels = axis.get_xticklabels()
    for label in ticklabels:
        label._text = label._text.split("/")[-1].replace("_", " ")
    axis.set_xticklabels(ticklabels,
                         **XTICKLABEL_PARAMS)
    axis.tick_params(**TICKPARAMS_PARAMS)
    axis.set_xlabel("")
    axis.set_ylabel(f"{SCORE_LABEL}", fontsize = AXIS_LABEL_SIZE)
    axis.set_ylim(SCORING_YLIMS)
    axis.set_title(f"{SCORE_LABEL} per gate", fontsize = TITLE_SIZE)
    axis = fig.add_subplot(axis)
    return

def _read_gate_freqs(dataset_name):
    return pd.read_csv(f"gate_counts/{dataset_name}_freqs.csv")

def _read_gate_counts(dataset_name):
    return pd.read_csv(f"gate_counts/{dataset_name}_counts.csv")

def _generate_score_heatmap_full(fig: Figure,
                                 ax: Axes,
                                 gs: GridSpec,
                                 subfigure_label: str,
                                 dataset_name: str,
                                 gates_to_use: str):

    gate_freqs = _read_gate_freqs(dataset_name = dataset_name)
    freqs = pd.DataFrame(gate_freqs.mean(), columns = ["freq"])
    freqs.index = [gate.replace("_,_", "_") for gate in freqs.index]

    gate_counts = _read_gate_counts(dataset_name = dataset_name)
    counts = pd.DataFrame(gate_counts.mean(), columns = ["freq"])
    counts.index = [gate.replace("_,_", "_") for gate in counts.index]

    tmp = pd.DataFrame()
    for assay in ["hyper_DT", "hyper_ET", "hyper_ETS", "hyper_RF", "hyper_MLP", "hyper_KNN"]:
        _remove_gate_comma_in_file(f"figure_data/Figure_2/{dataset_name}/{assay}/Scores.log")
        hyper_data = pd.read_csv(f"figure_data/Figure_2/{dataset_name}/{assay}/Scores.log", index_col = False)
        hyper_data = hyper_data[hyper_data["score_on"] == "val"]
        hyper_data["_sampling"] = hyper_data["sampling"].map({True: "sampled", False: "unsampled"})
        hyper_data["_tuned"] = hyper_data["tuned"].map({True: "tuned", False: "not tuned"})
        hyper_data["modification"] = [" + ".join([sampling, tuning])
                                      for sampling, tuning in
                                      zip(hyper_data["_sampling"], hyper_data["_tuned"])]
        tmp = pd.concat([tmp, hyper_data], axis = 0)

    hyper_data = tmp
    hyper_data = hyper_data[~hyper_data["gate"].str.contains("asd")]
    hyper_data.reset_index(drop = True)
    hyper_data.fillna(0)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    gates_to_use = hyper_data["gate"].unique()
    gates_to_use = [gate for gate in gates_to_use if not "Q" in gate]
    gates_to_use = gates_to_use[:min(30, len(gates_to_use))]


    hyper_data = hyper_data[hyper_data["gate"].isin(gates_to_use)]
    hyper_data["gate"] = [gate.split("/")[-1] +
                          f" (frac pos: {round(freqs.loc[freqs.index == gate,'freq'].iloc[0], 4)}; " +
                          f"pos. evts.: {int(counts.loc[counts.index == gate,'freq'].iloc[0])})"
                          for gate in hyper_data["gate"]]
    
    hyper_data = hyper_data[hyper_data["sampling"] == False]
    df = hyper_data.groupby(["algorithm", "train_size", "tuned", "gate"]).median("f1_score").reset_index()
    df = df.pivot(index='gate', columns=["algorithm", "train_size", "tuned"], values='f1_score')
    df = df.T.reset_index()

    metadata_annotation = ["algorithm", "train_size", "tuned"]
    gates = hyper_data["gate"].unique()
    col_colors= [
        _map_obs_to_cmap(df, group, ANNOTATION_CMAPS[i])
                         for i, group in enumerate(metadata_annotation)
    ]
    clustermap = sns.clustermap(df[gates].T,
                                cmap = "YlOrRd",
                                col_colors = col_colors,
                                row_cluster = False,
                                col_cluster = False,
                                figsize = (FIGURE_WIDTH_FULL, 6),
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
    clustermap.savefig('f1_score_heatmap.png', dpi = 300,
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

def _generate_score_heatmap_sampling(fig: Figure,
                                     ax: Axes,
                                     gs: GridSpec,
                                     subfigure_label: str,
                                     dataset_name: str,
                                     gates_to_use: str):

    gate_freqs = _read_gate_freqs(dataset_name = dataset_name)
    freqs = pd.DataFrame(gate_freqs.mean(), columns = ["freq"])
    freqs.index = [gate.replace("_,_", "_") for gate in freqs.index]

    gate_counts = _read_gate_counts(dataset_name = dataset_name)
    counts = pd.DataFrame(gate_counts.mean(), columns = ["freq"])
    counts.index = [gate.replace("_,_", "_") for gate in counts.index]

    tmp = pd.DataFrame()
    for assay in ["hyper_DT", "hyper_ET", "hyper_ETS", "hyper_RF", "hyper_MLP", "hyper_KNN"]:
        _remove_gate_comma_in_file(f"figure_data/Figure_2/{dataset_name}/{assay}/Scores.log")
        hyper_data = pd.read_csv(f"figure_data/Figure_2/{dataset_name}/{assay}/Scores.log", index_col = False)
        hyper_data = hyper_data[hyper_data["score_on"] == "val"]
        hyper_data["_sampling"] = hyper_data["sampling"].map({True: "sampled", False: "unsampled"})
        hyper_data["_tuned"] = hyper_data["tuned"].map({True: "tuned", False: "not tuned"})
        hyper_data["modification"] = [" + ".join([sampling, tuning])
                                      for sampling, tuning in
                                      zip(hyper_data["_sampling"], hyper_data["_tuned"])]
        tmp = pd.concat([tmp, hyper_data], axis = 0)

    hyper_data = tmp
    hyper_data = hyper_data[~hyper_data["gate"].str.contains("asd")]
    hyper_data.reset_index(drop = True)
    hyper_data.fillna(0)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    freqs = freqs.sort_values("freq", ascending = True)
    gates_to_use = [gate for gate in freqs.index if 0 < freqs.loc[freqs.index == gate, "freq"].iloc[0] < 0.005]
    gates_to_use = gates_to_use[:min(20, len(gates_to_use))]

    hyper_data = hyper_data[hyper_data["gate"].isin(gates_to_use)]
    hyper_data["gate"] = [gate.split("/")[-1] +
                          f" (frac pos: {round(freqs.loc[freqs.index == gate,'freq'].iloc[0], 4)}; " +
                          f"pos. evts.: {int(counts.loc[counts.index == gate,'freq'].iloc[0])})"
                          for gate in hyper_data["gate"]]
    
    hyper_data = hyper_data[hyper_data["tuned"] == True]
    df = hyper_data.groupby(["algorithm", "gate", "sampling"]).median("recall_score").reset_index()
    df = df.pivot(index='gate', columns=["algorithm", "sampling"], values='recall_score')
    df = df.T.reset_index()

    metadata_annotation = ["algorithm", "sampling"]
    gates = hyper_data["gate"].unique()
    col_colors= [
        _map_obs_to_cmap(df, group, ANNOTATION_CMAPS[i])
                         for i, group in enumerate(metadata_annotation)
    ]
    clustermap = sns.clustermap(df[gates].T,
                                cmap = "YlOrRd",
                                col_colors = col_colors,
                                row_cluster = False,
                                col_cluster = False,
                                figsize = (FIGURE_WIDTH_FULL, min(1*len(gates_to_use), 6)),
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
    clustermap.ax_cbar.set_xlabel("Recall score", fontsize = SCORELABEL_FONTSIZE)
    clustermap.savefig('f1_score_heatmap.png', dpi = 300,
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

def generate_classifier_characterization_heatmap(dataset_name: Literal[
                                                 "mouse_lineages_bm",
                                                 "mouse_lineages_spl",
                                                 "mouse_lineages_pb",
                                                 "HIMC",
                                                 "ZPM",
                                                 "human_t_cells",
                                                 "OMIP"
                                                 ],
                                              gates_to_use: list[str] = None,
                                              save: str = None,
                                              show: bool = True):

    fig = plt.figure(layout = "constrained",
                     figsize = (FIGURE_WIDTH_FULL, FIGURE_HEIGHT_FULL))
    gs = GridSpec(ncols = 1,
                  nrows = 2,
                  figure = fig,
                  height_ratios = [1, 1])
    
    a_coords = gs[0, 0]
    b_coords = gs[1, 0]

    fig_a = fig.add_subplot(a_coords)
    fig_b = fig.add_subplot(b_coords)

    _generate_score_heatmap_full(fig = fig,
                                 ax = fig_a,
                                 gs = a_coords,
                                 subfigure_label = "A",
                                 dataset_name = dataset_name,
                                 gates_to_use = gates_to_use)

    _generate_score_heatmap_sampling(fig = fig,
                                     ax = fig_b,
                                     gs = b_coords,
                                     subfigure_label = "B",
                                     dataset_name = dataset_name,
                                     gates_to_use = gates_to_use)
 
 
    if save:
        plt.savefig(os.path.join(os.getcwd(), save),
                    dpi = DPI,
                    bbox_inches = "tight")
        
    if show:
        plt.show()



def generate_classifier_characterization_fine(dataset_name: Literal[
                                                 "mouse_lineages_bm",
                                                 "mouse_lineages_spl",
                                                 "mouse_lineages_pb",
                                                 "HIMC",
                                                 "ZPM",
                                                 "human_t_cells",
                                                 "OMIP"
                                                 ],
                                              gates_to_use: list[str],
                                              save: str = None,
                                              show: bool = True):

    fig = plt.figure(layout = "constrained",
                     figsize = (FIGURE_WIDTH_FULL, FIGURE_HEIGHT_FULL))
    gs = GridSpec(ncols = 16,
                  nrows = 4,
                  figure = fig,
                  height_ratios = [1, 0.7, 0.7, 0.7])
    
    a_coords = gs[0, :8]
    b_coords = gs[0, 8:]

    c_coords = gs[1, :5]
    d_coords = gs[1, 5:8]
    e_coords = gs[1, 8:13]
    f_coords = gs[1, 13:16]

    g_coords = gs[2, :5]
    h_coords = gs[2, 5:8]
    i_coords = gs[2, 8:13]
    j_coords = gs[2, 13:16]

    k_coords = gs[3, :5]
    l_coords = gs[3, 5:8]
    m_coords = gs[3, 8:13]
    n_coords = gs[3, 13:16]

    fig_a = fig.add_subplot(a_coords)
    fig_b = fig.add_subplot(b_coords)

    fig_c = fig.add_subplot(c_coords)
    fig_d = fig.add_subplot(d_coords)
    fig_e = fig.add_subplot(e_coords)
    fig_f = fig.add_subplot(f_coords)

    fig_g = fig.add_subplot(g_coords)
    fig_h = fig.add_subplot(h_coords)
    fig_i = fig.add_subplot(i_coords)
    fig_j = fig.add_subplot(j_coords)

    fig_k = fig.add_subplot(k_coords)
    fig_l = fig.add_subplot(l_coords)
    fig_m = fig.add_subplot(m_coords)
    fig_n = fig.add_subplot(n_coords)

    _generate_classifier_overview(fig = fig,
                                  ax = fig_a,
                                  gs = a_coords,
                                  subfigure_label = "A",
                                  dataset_name = dataset_name,
                                  gates_to_use = gates_to_use)
    _generate_time_comparison(fig = fig,
                              ax = fig_b,
                              gs = b_coords,
                              subfigure_label = "B",
                              dataset_name = dataset_name)

    _generate_train_size_comparison(fig = fig,
                                    ax = fig_c,
                                    gs = c_coords,
                                    subfigure_label = "C",
                                    dataset_name = dataset_name,
                                    classifier = "RandomForestClassifier",
                                    gates_to_use = gates_to_use)
    _generate_hyperparameter_tuning_plot(fig = fig,
                                         ax = fig_d,
                                         gs = d_coords,
                                         subfigure_label = "D",
                                         dataset_name = dataset_name,
                                         classifier = "RandomForestClassifier")

    _generate_train_size_comparison(fig = fig,
                                    ax = fig_e,
                                    gs = e_coords,
                                    subfigure_label = "E",
                                    dataset_name = dataset_name,
                                    classifier = "DecisionTreeClassifier",
                                    gates_to_use = gates_to_use)
    _generate_hyperparameter_tuning_plot(fig = fig,
                                         ax = fig_f,
                                         gs = f_coords,
                                         subfigure_label = "F",
                                         dataset_name = dataset_name,
                                         classifier = "DecisionTreeClassifier")


    _generate_train_size_comparison(fig = fig,
                                    ax = fig_g,
                                    gs = g_coords,
                                    subfigure_label = "G",
                                    dataset_name = dataset_name,
                                    classifier = "ExtraTreesClassifier",
                                    gates_to_use = gates_to_use)
    _generate_hyperparameter_tuning_plot(fig = fig,
                                         ax = fig_h,
                                         gs = h_coords,
                                         subfigure_label = "H",
                                         dataset_name = dataset_name,
                                         classifier = "ExtraTreesClassifier")


    _generate_train_size_comparison(fig = fig,
                                    ax = fig_i,
                                    gs = i_coords,
                                    subfigure_label = "I",
                                    dataset_name = dataset_name,
                                    classifier = "ExtraTreeClassifier",
                                    gates_to_use = gates_to_use)
    _generate_hyperparameter_tuning_plot(fig = fig,
                                         ax = fig_j,
                                         gs = j_coords,
                                         subfigure_label = "J",
                                         dataset_name = dataset_name,
                                         classifier = "ExtraTreeClassifier")


    _generate_train_size_comparison(fig = fig,
                                    ax = fig_k,
                                    gs = k_coords,
                                    subfigure_label = "K",
                                    dataset_name = dataset_name,
                                    classifier = "MLPClassifier",
                                    gates_to_use = gates_to_use)
    _generate_hyperparameter_tuning_plot(fig = fig,
                                         ax = fig_l,
                                         gs = l_coords,
                                         subfigure_label = "L",
                                         dataset_name = dataset_name,
                                         classifier = "MLPClassifier")


    _generate_train_size_comparison(fig = fig,
                                    ax = fig_m,
                                    gs = m_coords,
                                    subfigure_label = "M",
                                    dataset_name = dataset_name,
                                    classifier = "KNN",
                                    gates_to_use = gates_to_use)
    _generate_hyperparameter_tuning_plot(fig = fig,
                                         ax = fig_n,
                                         gs = n_coords,
                                         subfigure_label = "N",
                                         dataset_name = dataset_name,
                                         classifier = "KNN")

    if save:
        plt.savefig(os.path.join(os.getcwd(), save),
                    dpi = DPI,
                    bbox_inches = "tight")
        
    if show:
        plt.show()

    return