import os
import pickle

from anndata import AnnData
import numpy as np
import pandas as pd
import scanpy as sc
import FACSPy as fp

from FACSPy._utils import (_find_parent_gate,
                           _find_gate_path_of_gate,
                           _create_gate_lut)

import seaborn as sns
import matplotlib.patches as patches
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.patches import ConnectionPatch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from scipy.sparse import csr_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin

from typing import Literal, Union, Optional

from ._utils import (_figure_label,
                     _remove_gate_comma_in_file,
                     _generate_dataset,
                     _scanpy_vector_friendly,
                     _remove_underscores_from_gates_ticklabels,
                     _remove_underscores_from_gates,
                     _extract_population_from_gates,
                     BOXPLOT_PARAMS,
                     STRIPPLOT_PARAMS,
                     XTICKLABEL_PARAMS,
                     TICKPARAMS_PARAMS,
                     CENTERED_LEGEND_PARAMS,
                     FIGURE_HEIGHT_FULL,
                     FIGURE_WIDTH_FULL,
                     TITLE_SIZE,
                     AXIS_LABEL_SIZE,
                     SUPERVISED_SCORE,
                     UMAP_LABEL_SIZE,
                     SCORING_YLIMS,
                     DPI)
from ..utils.classifier_list import CLASSIFIERS_TO_TEST

SCORE_LABEL = SUPERVISED_SCORE.replace("_", " ")

CLASSIFIERS_TO_SHOW = ["RandomForestClassifier", "DecisionTreeClassifier", "ExtraTreesClassifier", "ExtraTreeClassifier", "MLPClassifier", "KNN"]

RAW_INPUT_DIR = os.path.join(os.getcwd(), "datasets/")
DATASET_DIR = os.path.join(os.getcwd(), "figure_data/Figure_2")

def _read_score_data(dataset_name: str,
                     gates_to_use: list[str]):
    score_data_path = os.path.join(DATASET_DIR,
                                   f"{dataset_name}/classifier_comparison/Scores.log")
    _remove_gate_comma_in_file(score_data_path)
    
    score_data = pd.read_csv(score_data_path,
                             index_col = False)
    score_data = score_data[score_data["score_on"] == "val"]
    score_data = score_data[score_data["gate"].isin(gates_to_use)]
    score_data = score_data.fillna(0)
    return score_data

def _read_technical_data(dataset_name: str):
    technical_data = pd.read_csv(os.path.join(DATASET_DIR,
                                          f"{dataset_name}/classifier_comparison/Technicals.log"),
                                 index_col = False)
    technical_data["mem_used"] = technical_data["max_mem"] - technical_data["min_mem"]
    return technical_data

def _extract_fops(orig_df: pd.DataFrame,
                  pred_df: pd.DataFrame,
                  gate: str,
                  dataset: AnnData):
    orig_df = orig_df.reset_index()
    pred_df = pred_df.reset_index()
    gate_path = _find_gate_path_of_gate(dataset, gate)
    parent_gate_path = _find_parent_gate(gate_path)
    orig_df = orig_df[orig_df["gate"] == gate_path]
    orig_df = orig_df[orig_df["freq_of"] == parent_gate_path]
    orig_df["gating"] = "Expert"
    
    pred_df = pred_df[pred_df["gate"] == gate_path]
    pred_df = pred_df[pred_df["freq_of"] == parent_gate_path]
    pred_df["gating"] = "AI"
    
    return pd.concat([orig_df, pred_df], axis = 0)

def _add_gates_to_plot(ax: Axes,
                       gate_lut: dict,
                       gates: Union[str, list[str]],
                       linestyle: str = "line",
                       linecolor: str = "red") -> Axes:
    
    gate_line_params = {
        "marker": ".",
        "markersize": 2,
        "color": linecolor,
        "linestyle": "-" if linestyle == "line" else "--"
    }
    hvline_params = {
        "color": "red",
        "linestyle": "-" if linestyle == "line" else "--"
    }
    if not isinstance(gates, list):
        gates = [gates]
    for gate in gates:
        gate_dict = gate_lut[gate]
        vertices = gate_dict["vertices"]
        if gate_dict["gate_type"] == "PolygonGate":
            ax.plot(vertices[:,0],
                    vertices[:,1],
                    **gate_line_params)
        elif gate_dict["gate_type"] == "GMLRectangleGate":
            ### handles the case of quandrant gates
            if np.isnan(vertices).any():
                if any(np.isnan(vertices[:,0])):
                    if all(np.isnan(vertices[:,0])):
                        ax.axvline(x = np.nan,
                                   **hvline_params)
                    else:
                        #TODO incredibly messy...
                        ax.axvline(x = int(vertices[0][~np.isnan(vertices[0])][0]),
                                   **hvline_params)
                if any(np.isnan(vertices[:,1])):
                    if all(np.isnan(vertices[:,1])):
                        ax.axhline(y = np.nan,
                                **hvline_params)
                    else:
                        #TODO incredibly messy...
                        ax.axhline(y = int(vertices[1][~np.isnan(vertices[1])][0]),
                            **hvline_params)              
                continue
            
            patch_starting_point = (vertices[0,0], vertices[1,0])
            height = abs(int(np.diff(vertices[1])))
            width = abs(int(np.diff(vertices[0])))
            ax.add_patch(
                patches.Rectangle(
                    xy = patch_starting_point,
                    width = width,
                    height = height,
                    facecolor = "none",
                    edgecolor = linecolor,
                    linestyle = "-" if linestyle == "line" else "--",
                    linewidth = 1
                )
            )
            ax = _adjust_viewlim(ax, patch_starting_point, height, width)

    return ax

def _adjust_viewlim(ax: Axes,
                    patch_starting_point: tuple[float, float],
                    height: int,
                    width: int) -> Axes:
    current_viewlim = ax.viewLim
    current_x_lims = current_viewlim._points[:,0]
    current_y_lims = current_viewlim._points[:,1]
    x0 = calculate_range_extension_viewLim(point = min(current_x_lims[0], patch_starting_point[0]),
                                           point_loc = "min")
    y0 = calculate_range_extension_viewLim(point = min(current_y_lims[0], patch_starting_point[1]),
                                           point_loc = "min")
    x1 = calculate_range_extension_viewLim(point = max(current_x_lims[1], patch_starting_point[0] + width),
                                           point_loc = "max")
    y1 = calculate_range_extension_viewLim(point = max(current_y_lims[1], patch_starting_point[1] + height),
                                           point_loc = "max")
    current_viewlim.set_points(np.array([[x0,y0],
                                         [x1, y1]]))
    return ax

def calculate_range_extension_viewLim(point: float,
                                      point_loc: Literal["min", "max"]) -> float:
    if point_loc == "min":
        return point * 1.1 if point < 0 else point * 0.9
    if point_loc == "max":
        return point * 0.9 if point < 0 else point * 1.1
    
def extract_gate_lut(adata: AnnData,
                     wsp_group: str,
                     file_name: str) -> dict[str: dict[str: Union[list[str], str]]]:
    return _create_gate_lut(adata.uns["workspace"][wsp_group])[file_name]

    
def _generate_figure_a(fig: Figure,
                       ax: Axes,
                       gs: GridSpec,
                       subfigure_label: str,
                       dataset: AnnData,
                       palette: str,
                       biax_kwargs: dict,
                       biax_layout_kwargs: dict,
                       gate_lut: dict,
                       graphical_abstract_gate: str) -> None:
    ax.axis("off")
    _figure_label(ax, subfigure_label, x = -0.24)
    fig_sgs = gs.subgridspec(2,2)

    empty_umap = fig.add_subplot(fig_sgs[0,0])

    _scanpy_vector_friendly()
    empty_umap_plot: Axes = fp.pl.umap(
        dataset,
        color = None,
        show = False,
        ax = empty_umap
    )
    empty_umap.set_xlabel(empty_umap_plot.get_xlabel(), fontsize = AXIS_LABEL_SIZE)
    empty_umap.set_ylabel(empty_umap_plot.get_ylabel(), fontsize = AXIS_LABEL_SIZE)
    empty_umap.set_title("Ungated cells", fontsize = TITLE_SIZE)
    empty_umap = fig.add_subplot(empty_umap_plot)
    sns.reset_defaults()

    biax_gated = fig.add_subplot(fig_sgs[0,1])
    biax_gated_plot = fp.pl.biax(
        dataset,
        ax = biax_gated,
        **biax_kwargs
    )

    biax_gated_plot = _add_gates_to_plot(ax = biax_gated,
                                         gate_lut = gate_lut,
                                         gates = graphical_abstract_gate)
    biax_gated_plot.set_xlim(biax_layout_kwargs["xlim_0"], biax_gated.get_xlim()[1])
    biax_gated_plot.tick_params(axis='both', labelsize = AXIS_LABEL_SIZE,
                                labelbottom = False, bottom = False, left = False, labelleft = False)
    biax_gated.set_xlabel(biax_gated_plot.get_xlabel(), fontsize = AXIS_LABEL_SIZE)
    biax_gated.set_ylabel(biax_gated_plot.get_ylabel(), fontsize = AXIS_LABEL_SIZE)
    biax_gated.set_title("Training on\ngated examples", fontsize = TITLE_SIZE)
    biax_gated = fig.add_subplot(
        biax_gated_plot
    )

    biax_predicted = fig.add_subplot(fig_sgs[1,0])
    new_sample_ID = int(biax_kwargs["sample_identifier"])+1
    if not str(new_sample_ID) in dataset.obs["sample_ID"].unique():
        new_sample_ID += 1

    biax_kwargs["sample_identifier"] = str(new_sample_ID)
    biax_predicted_plot = fp.pl.biax(
        dataset,
        ax = biax_predicted,
        **biax_kwargs
    )
    biax_predicted_plot = _add_gates_to_plot(ax = biax_predicted,
                                             gate_lut = gate_lut,
                                             gates = graphical_abstract_gate,
                                             linestyle = "dashed")
    biax_predicted_plot.set_xlim(biax_layout_kwargs["xlim_0"], biax_predicted.get_xlim()[1])
    biax_predicted_plot.tick_params(axis='both', labelsize = AXIS_LABEL_SIZE, labelbottom = False, bottom = False, left = False, labelleft = False)
    biax_predicted.set_xlabel(biax_predicted_plot.get_xlabel(), fontsize = AXIS_LABEL_SIZE)
    biax_predicted.set_ylabel(biax_predicted_plot.get_ylabel(), fontsize = AXIS_LABEL_SIZE)
    biax_predicted.set_title("Prediction on\nungated samples", fontsize = TITLE_SIZE)
    biax_predicted = fig.add_subplot(
        biax_predicted_plot
    )
    cell_type_annotation = fig.add_subplot(fig_sgs[1,1])

    _scanpy_vector_friendly()
    cell_type_annotation_plot: Axes = fp.pl.umap(
        dataset,
        color = graphical_abstract_gate,
        show = False,
        ax = cell_type_annotation,
        s = 2,
        cmap = palette,
        legend_loc = "on data",
        legend_fontsize = AXIS_LABEL_SIZE
    )
    handles, labels = cell_type_annotation.get_legend_handles_labels()
    cell_type_annotation.legend(handles, labels,
                                **CENTERED_LEGEND_PARAMS)
    cell_type_annotation_plot.set_title("cell\nannotation", fontsize = 8)
    cell_type_annotation_plot.set_xlabel(cell_type_annotation_plot.get_xlabel(), fontsize = AXIS_LABEL_SIZE)
    cell_type_annotation_plot.set_ylabel(cell_type_annotation_plot.get_ylabel(), fontsize = AXIS_LABEL_SIZE)
    cell_type_annotation_plot.legend().remove()
    cell_type_annotation = fig.add_subplot(
        cell_type_annotation_plot
    )
    sns.reset_defaults()
    
    con1 = ConnectionPatch(xyA=(empty_umap_plot.get_xlim()[1],
                            empty_umap_plot.get_ylim()[0]), coordsA=empty_umap.transData, 
                       xyB=(biax_gated_plot.get_xlim()[0],
                            biax_gated_plot.get_ylim()[0]), coordsB=biax_gated.transData,
                       color = 'black',
                       arrowstyle = "->")
    fig.add_artist(con1)

    con2 = ConnectionPatch(xyA=(biax_gated_plot.get_xlim()[0],
                                biax_gated_plot.get_ylim()[0]), coordsA=biax_gated.transData, 
                           xyB=(biax_predicted_plot.get_xlim()[1],
                                biax_predicted_plot.get_ylim()[1]), coordsB=biax_predicted.transData,
                           color = 'black',
                           arrowstyle = "->")
    fig.add_artist(con2)

    con3 = ConnectionPatch(xyA=(biax_predicted_plot.get_xlim()[1],
                                biax_predicted_plot.get_ylim()[1]), coordsA=biax_predicted.transData,
                           xyB=(cell_type_annotation_plot.get_xlim()[0],
                                cell_type_annotation_plot.get_ylim()[1]), coordsB=cell_type_annotation.transData,
                           color = 'black',
                           arrowstyle = "->")
    fig.add_artist(con3)

    return

def _generate_figure_b(fig: Figure,
                       ax: Axes,
                       gs: GridSpec,
                       subfigure_label: str,
                       score_data: pd.DataFrame) -> None:
    
    ax.axis("off")
    _figure_label(ax, subfigure_label, x = -0.2)
    fig_sgs = gs.subgridspec(1,1)

    score_data = score_data[score_data["algorithm"].isin(CLASSIFIERS_TO_SHOW)]

    axis = fig.add_subplot(fig_sgs[0])
    plot_kwargs = {
        "data": score_data,
        "x": "algorithm",
        "y": SUPERVISED_SCORE,
        "hue": "gate",
        "ax": axis
    }
    sns.stripplot(**plot_kwargs,
                  **STRIPPLOT_PARAMS)
    sns.boxplot(**plot_kwargs,
                **BOXPLOT_PARAMS)
    handles, _ = axis.get_legend_handles_labels()

    gates = _extract_population_from_gates(score_data["gate"].unique())
    gates = _remove_underscores_from_gates(gates)
    axis.legend(handles, gates,
                **CENTERED_LEGEND_PARAMS)
    ticklabels = [clf.split("Classifier")[0] for clf in score_data["algorithm"].unique()]
    axis.set_xticklabels(ticklabels,
                         **XTICKLABEL_PARAMS)
    axis.tick_params(**TICKPARAMS_PARAMS)
    axis.set_xlabel("")
    axis.set_ylabel(SCORE_LABEL, fontsize = AXIS_LABEL_SIZE)
    axis.set_ylim(SCORING_YLIMS)
    axis.set_title("Classifier Comparison", fontsize = TITLE_SIZE)

    axis = fig.add_subplot(axis)
    return

def _generate_figure_c(fig: Figure,
                       ax: Axes,
                       gs: GridSpec,
                       subfigure_label: str,
                       technical_data: pd.DataFrame) -> None:
    
    ax.axis("off")
    _figure_label(ax, subfigure_label, x = -0.5)
    fig_sgs = gs.subgridspec(1,1)
    axis = fig.add_subplot(fig_sgs[0])
    technical_data = technical_data[technical_data["algorithm"].isin(CLASSIFIERS_TO_SHOW)]
    plot_kwargs = {
        "data": technical_data,
        "x": "algorithm",
        "y": "train_time",
        "ax": axis
    }
    sns.stripplot(**plot_kwargs,
                  **STRIPPLOT_PARAMS)
    sns.boxplot(**plot_kwargs,
                **BOXPLOT_PARAMS)
    ticklabels = [clf.split("Classifier")[0] for clf in technical_data["algorithm"].unique()]
    axis.set_xticklabels(ticklabels,
                         **XTICKLABEL_PARAMS)
    axis.tick_params(**TICKPARAMS_PARAMS)
    axis.set_xlabel("")
    axis.set_ylabel("classification time\nper sample [s]",
                    fontsize = AXIS_LABEL_SIZE)
    axis.set_ylim(0, axis.get_ylim()[1])
    axis.set_title("classification time", fontsize = TITLE_SIZE)
    axis = fig.add_subplot(axis)

    return

def _generate_figure_d(fig: Figure,
                       ax: Axes,
                       gs: GridSpec,
                       subfigure_label: str,
                       clf_data: pd.DataFrame) -> None:
    
    ax.axis("off")
    _figure_label(ax, subfigure_label, x = -0.5)
    fig_sgs = gs.subgridspec(1,1)

    clf_data["gate"] = [gate.split("/")[-1] for gate in clf_data["gate"].tolist()]

    axis = fig.add_subplot(fig_sgs[0])
    plot_kwargs = {
        "data": clf_data,
        "x": "gate",
        "y": SUPERVISED_SCORE,
        "ax": axis
    }
    sns.stripplot(**plot_kwargs,
                  **STRIPPLOT_PARAMS)
    sns.boxplot(**plot_kwargs,
                **BOXPLOT_PARAMS)
    
    ticklabels = axis.get_xticklabels()
    ticklabels = _remove_underscores_from_gates_ticklabels(ticklabels)
    
    axis.set_xticklabels(ticklabels,
                         **XTICKLABEL_PARAMS)
    axis.tick_params(**TICKPARAMS_PARAMS)
    axis.set_xlabel("")
    axis.set_ylabel(SCORE_LABEL, fontsize = AXIS_LABEL_SIZE)
    axis.set_ylim(SCORING_YLIMS)
    axis.set_title(f"{SCORE_LABEL} per gate", fontsize = TITLE_SIZE)
    axis = fig.add_subplot(axis)

    return
    
def _generate_figure_e(fig: Figure,
                       ax: Axes,
                       gs: GridSpec,
                       subfigure_label: str,
                       orig_gf: pd.DataFrame,
                       pred_gf: pd.DataFrame,
                       gates_to_use: list[str],
                       dataset: AnnData) -> None:
    
    ax.axis("off")
    _figure_label(ax, subfigure_label, x = -0.2)
    fig_sgs = gs.subgridspec(3,3)
    col_coord = -1
    for i, gate in enumerate(gates_to_use):
        df = _extract_fops(orig_gf, pred_gf, gate.split("/")[-1], dataset)
        row_coord = i%3
        if row_coord == 0:
            col_coord += 1
        comp_plot = fig.add_subplot(fig_sgs[col_coord, row_coord])
        plot_kwargs = {
            "data": df,
            "x": "gating",
            "y": "freq",
            "ax": comp_plot
        }
        sns.stripplot(**plot_kwargs,
                      **STRIPPLOT_PARAMS)
        sns.boxplot(**plot_kwargs,
                    **BOXPLOT_PARAMS)
        g = sns.pointplot(**plot_kwargs,
                          hue = "sample_ID",
                          color = "black",
                          dodge = 0.05,
                          join = True,
                          markers = "",
                          linestyles = "dotted",
                          scale = 0.5)
        g.legend_.remove()

        gate = _extract_population_from_gates(gate)
        gate = _remove_underscores_from_gates(gate)
        comp_plot.set_title(gate, fontsize = TITLE_SIZE)
        comp_plot.set_xlabel("")
        comp_plot.set_ylabel("freq. of parent", fontsize = AXIS_LABEL_SIZE)
        comp_plot.tick_params(**TICKPARAMS_PARAMS)
        comp_plot = fig.add_subplot(comp_plot)

    return
    
def _generate_figure_f(fig: Figure,
                       ax: Axes,
                       gs: GridSpec,
                       subfigure_label: str,
                       orig_vis: AnnData,
                       pred_vis: AnnData,
                       gates_to_use: list[str]) -> None:
    
    ax.axis("off")
    _figure_label(ax, subfigure_label, x = -0.1)
    n_plots = 8
    fig_sgs = gs.subgridspec(2,n_plots)

    umap_plot_kwargs = {
        "legend_loc": None,
        "legend_fontsize": UMAP_LABEL_SIZE,
        "cmap": "Set1",
        "s": 1,
        "legend_fontoutline": 1
    }
    cell_types = [gate.split("/")[-1] for gate in gates_to_use][:n_plots]
    _scanpy_vector_friendly()
    for i, cell_type in enumerate(cell_types):
        gating_umap = fig.add_subplot(fig_sgs[0,i])
        gating_umap_plot: Axes = fp.pl.umap(
            orig_vis,
            color = cell_type,
            ax = gating_umap,
            show = False,
            **umap_plot_kwargs
        )
        gating_umap_plot.set_title(f"{_remove_underscores_from_gates(cell_type)}\nExpert", fontsize = TITLE_SIZE)
        gating_umap_plot.set_xlabel(gating_umap_plot.get_xlabel(), fontsize = AXIS_LABEL_SIZE)
        gating_umap_plot.set_ylabel(gating_umap_plot.get_ylabel(), fontsize = AXIS_LABEL_SIZE)
        gating_umap = fig.add_subplot(gating_umap_plot)

        gating_umap = fig.add_subplot(fig_sgs[1,i])
        gating_umap_plot = fp.pl.umap(
            pred_vis,
            color = cell_type,
            ax = gating_umap,
            show = False,
            **umap_plot_kwargs
        )
        gating_umap_plot.set_title("FACSPy", fontsize = TITLE_SIZE)
        gating_umap_plot.set_xlabel(gating_umap_plot.get_xlabel(), fontsize = AXIS_LABEL_SIZE)
        gating_umap_plot.set_ylabel(gating_umap_plot.get_ylabel(), fontsize = AXIS_LABEL_SIZE)
        gating_umap = fig.add_subplot(gating_umap_plot)
    sns.reset_defaults()
    return

def _train_classifier(clf: ClassifierMixin,
                      X_train: np.ndarray,
                      y_train: np.ndarray) -> ClassifierMixin:
    return clf.fit(X_train, y_train)

def _get_best_params(classifier: str,
                     dataset_name: str):
    if classifier == "DecisionTreeClassifier":
        best_params_folder = "hyper_DT"
    elif classifier == "ExtraTreeClassifier":
        best_params_folder = "hyper_ET"
    elif classifier == "RandomForestClassifier":
        best_params_folder = "hyper_RF"
    elif classifier == "ExtraTreesClassifier":
        best_params_folder = "hyper_ETS"
    elif classifier == "KNN":
        best_params_folder = "hyper_KNN"
    elif classifier == "MLPClassifier":
        best_params_folder = "hyper_MLP"
    
    with open(os.path.join(DATASET_DIR, f"{dataset_name}/{best_params_folder}/best_params/best_params_{classifier}_unsampled.dict"), "rb") as file:
        best_params_ = pickle.load(file)
    
    return best_params_


def _generate_visualization_datasets(dataset_name: str,
                                     classifier: str,
                                     train_sample_IDs: list[str],
                                     population_to_show: str,
                                     gates_to_use: list[str],
                                     palette: str = "Set1",
                                     pca_kwargs: Optional[dict] = None,
                                     umap_kwargs: Optional[dict] = None,
                                     neighbors_kwargs: Optional[dict] = None,
                                     return_data: bool = False):
    
    if pca_kwargs is None:
        pca_kwargs = {}
    if neighbors_kwargs is None:
        neighbors_kwargs = {}
    if umap_kwargs is None:
        umap_kwargs = {}

    dataset: AnnData = _generate_dataset(dataset_name,
                                         DATASET_DIR = DATASET_DIR,
                                         RAW_INPUT_DIR = RAW_INPUT_DIR)
    train_samples = dataset[dataset.obs["sample_ID"].isin(train_sample_IDs),:].copy()
    test_samples = dataset[~dataset.obs["sample_ID"].isin(train_sample_IDs),:].copy()
    scaler = StandardScaler()
    transformed_data = train_samples.layers["transformed"]
    print("Fitting scaler...")
    scaler.fit(transformed_data)

    classifier_dir = os.path.join(DATASET_DIR, f"{dataset_name}/{classifier}_trained.model")
    if os.path.isfile(classifier_dir):
        print("Loading pre-trained classifier...")
        with open(classifier_dir, "rb") as file:
            clf = pickle.load(file)
    else:
        best_params_ = _get_best_params(classifier, dataset_name)
        clf = CLASSIFIERS_TO_TEST[classifier]["classifier"](**best_params_)
        train_samples.layers["preprocessed"] = scaler.transform(transformed_data)
        X_train, _, y_train, _= train_test_split(train_samples.layers["preprocessed"],
                                                 train_samples.obsm["gating"].toarray(),
                                                 test_size = 0.1)
        print("Training classifier...")
        clf = _train_classifier(clf, X_train, y_train)
        print("... finished")
        with open(classifier_dir, "wb") as file:
            pickle.dump(clf, file)

    X_val = scaler.transform(test_samples.layers["transformed"])
    predicted_samples = test_samples.copy()
    predicted_samples.obsm["gating"] = csr_matrix(clf.predict(X_val))
    
    print("Calculating gate frequencies...")
    fp.tl.gate_frequencies(predicted_samples)
    fp.tl.gate_frequencies(test_samples)

    pred_gf: pd.DataFrame = predicted_samples.uns["gate_frequencies"]
    orig_gf: pd.DataFrame = test_samples.uns["gate_frequencies"]

    pred_gf.to_csv(os.path.join(DATASET_DIR, f"{dataset_name}/predicted_gating.csv"))
    orig_gf.to_csv(os.path.join(DATASET_DIR, f"{dataset_name}/original_gating.csv"))

    orig_vis = test_samples.copy()
    fp.subset_gate(orig_vis, population_to_show)
    sc.pp.subsample(orig_vis, n_obs = 10_000)

    print("Running PCA...")
    fp.tl.pca(orig_vis,
              **pca_kwargs)
    print("Calculating neighbors...")
    fp.tl.neighbors(orig_vis,
                    **neighbors_kwargs)
    print("Running UMAP...")
    fp.tl.umap(orig_vis,
               **umap_kwargs)
    
    pred_vis = predicted_samples[orig_vis.obs_names,:].copy()
    pred_vis.obsm[f"X_umap_{population_to_show}_transformed"] = orig_vis.obsm[f"X_umap_{population_to_show}_transformed"]

    color_palette = [colors.to_hex(color)
                     for color in sns.color_palette(palette)[:2]]
    for gate in gates_to_use:
        gate_name = gate.split("/")[-1]

        orig_vis.uns[f"{gate_name}_colors"] = color_palette
        fp.convert_gate_to_obs(orig_vis, gate_name)

        pred_vis.uns[f"{gate_name}_colors"] = color_palette
        fp.convert_gate_to_obs(pred_vis, gate_name)
    
    print("saving datasets...")
    fp.save_dataset(orig_vis,
                    output_dir = os.path.join(DATASET_DIR, dataset_name),
                    file_name = "visualization_original",
                    overwrite = True)
    fp.save_dataset(pred_vis,
                    output_dir = os.path.join(DATASET_DIR, dataset_name),
                    file_name = "visualization_predicted",
                    overwrite = True)
    
    print("\nVisualization Datasets Created\n")
    if return_data:
        return orig_vis, pred_vis, orig_gf, pred_gf
    else:
        return
    
def generate_supervised_characterization(dataset_name: Literal[
                                            "mouse_lineages_bm",
                                            "mouse_lineages_spl",
                                            "mouse_lineages_pb",
                                            "HIMC",
                                            "ZPM",
                                            "human_t_cells",
                                            "OMIP"
                                            ],
                                         population_to_show: str,
                                         gates_to_use: list[str],
                                         biax_kwargs: dict,
                                         biax_layout_kwargs: dict,
                                         wsp_group: str,
                                         wsp_file: str,
                                         graphical_abstract_gate: str,
                                         train_sample_IDs: list[str] = None,
                                         classifier: Literal["RandomForestClassifier",
                                                             "DecisionTreeClassifier",
                                                             "ExtraTreesClassifier",
                                                             "ExtraTreeClassifier",
                                                             "MLPClassifier",
                                                             "KNN"] = None,
                                         pca_kwargs: Optional[dict] = None,
                                         neighbors_kwargs: Optional[dict] = None,
                                         umap_kwargs: Optional[dict] = None,
                                         palette: str = "Set1",
                                         save: Optional[str] = None,
                                         show: bool = True
                                        ):
    fp.settings.default_gate = population_to_show
    fp.settings.default_layer = "transformed"

    visualizing_original_file = os.path.join(DATASET_DIR, f"{dataset_name}/visualization_original.h5ad")
    visualizing_predicted_file = os.path.join(DATASET_DIR, f"{dataset_name}/visualization_predicted.h5ad")

    if os.path.isfile(visualizing_original_file) and os.path.isfile(visualizing_predicted_file):
        print("Loading visualization datasets...")
        orig_vis = fp.read_dataset(os.path.dirname(visualizing_original_file),
                                   file_name = "visualization_original")
        pred_vis = fp.read_dataset(os.path.dirname(visualizing_predicted_file),
                                   file_name = "visualization_predicted")
        orig_gf = pd.read_csv(os.path.join(DATASET_DIR, f"{dataset_name}/original_gating.csv"))
        pred_gf = pd.read_csv(os.path.join(DATASET_DIR, f"{dataset_name}/predicted_gating.csv"))
    else:
        orig_vis, pred_vis, orig_gf, pred_gf = _generate_visualization_datasets(dataset_name = dataset_name,
                                                                                classifier = classifier,
                                                                                train_sample_IDs = train_sample_IDs,
                                                                                population_to_show = population_to_show,
                                                                                gates_to_use = gates_to_use,
                                                                                palette = palette,
                                                                                pca_kwargs = pca_kwargs,
                                                                                umap_kwargs = umap_kwargs,
                                                                                neighbors_kwargs = neighbors_kwargs,
                                                                                return_data = True)
    
    dataset = _generate_dataset(dataset_name = dataset_name,
                                RAW_INPUT_DIR = RAW_INPUT_DIR,
                                DATASET_DIR = DATASET_DIR)
    
    score_data = _read_score_data(dataset_name = dataset_name,
                                  gates_to_use = gates_to_use)
    clf_data = score_data[score_data["algorithm"] == classifier].copy()
    technical_data = _read_technical_data(dataset_name)

    fig = plt.figure(layout = "constrained",
                     figsize = (FIGURE_WIDTH_FULL, FIGURE_HEIGHT_FULL))
    gs = GridSpec(ncols = 6,
                  nrows = 4,
                  figure = fig,
                  height_ratios = [0.9, 0.75, 0.75, 0.7])
    
    a_coords = gs[0, :3]
    b_coords = gs[0, 2:]
    c_coords = gs[1, :2]
    d_coords = gs[2, :2]
    e_coords = gs[1:3, 2:6]
    f_coords = gs[3, :]

    fig_a = fig.add_subplot(a_coords)
    fig_b = fig.add_subplot(b_coords)
    fig_c = fig.add_subplot(c_coords)
    fig_d = fig.add_subplot(d_coords)
    fig_e = fig.add_subplot(e_coords)
    fig_f = fig.add_subplot(f_coords)

    gate_lut = extract_gate_lut(dataset, wsp_group, wsp_file)
    _generate_figure_a(fig = fig,
                       ax = fig_a,
                       gs = gs[0,:2],
                       subfigure_label = "A",
                       dataset = orig_vis,
                       palette = palette,
                       biax_kwargs = biax_kwargs,
                       biax_layout_kwargs = biax_layout_kwargs,
                       gate_lut = gate_lut,
                       graphical_abstract_gate = graphical_abstract_gate)
    _generate_figure_b(fig = fig,
                       ax = fig_b,
                       gs = b_coords,
                       subfigure_label = "B",
                       score_data = score_data)
    _generate_figure_c(fig = fig,
                       ax = fig_c,
                       gs = c_coords,
                       subfigure_label = "C",
                       technical_data = technical_data)
    _generate_figure_d(fig = fig,
                       ax = fig_d,
                       gs = d_coords,
                       subfigure_label = "D",
                       clf_data = clf_data)
    _generate_figure_e(fig = fig,
                       ax = fig_e,
                       gs = e_coords,
                       subfigure_label = "E",
                       orig_gf = orig_gf,
                       pred_gf = pred_gf,
                       gates_to_use = gates_to_use,
                       dataset = dataset)
    _generate_figure_f(fig = fig,
                       ax = fig_f,
                       gs = f_coords,
                       subfigure_label = "F",
                       orig_vis = orig_vis,
                       pred_vis = pred_vis,
                       gates_to_use = gates_to_use)

    if save:
        plt.savefig(os.path.join(os.getcwd(), save),
                    dpi = DPI,
                    bbox_inches = "tight")
        
    if show:
        plt.show()

    return