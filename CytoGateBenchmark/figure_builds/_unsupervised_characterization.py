import os

from anndata import AnnData
import numpy as np
import pandas as pd
import scanpy as sc
import FACSPy as fp

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.patches import ConnectionPatch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from typing import Literal, Optional

from ._utils import (_figure_label,
                     _generate_dataset,
                     _remove_gate_comma_in_file,
                     _get_gating_information,
                     _scanpy_vector_friendly,
                     _extract_population_from_gates,
                     _extract_population_from_gates_ticklabels,
                     _remove_underscores_from_gates_ticklabels,
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
                     UMAP_LABEL_SIZE,
                     SCORING_YLIMS,
                     DPI)

RAW_INPUT_DIR = os.path.join(os.getcwd(), "datasets/")
DATASET_DIR = os.path.join(os.getcwd(), "figure_data/Figure_3")

SCORE_LABEL = UNSUPERVISED_SCORE.replace("_", " ")

def _read_score_data(dataset_name: str,
                     gates_to_use: list[str]):
    score_data_path = os.path.join(DATASET_DIR,
                                   f"{dataset_name}/algorithm_comparison/Scores.log")
    _remove_gate_comma_in_file(score_data_path)
    
    score_data = pd.read_csv(score_data_path,
                             index_col = False)
    if dataset_name != "human_t_cells":
        gates_to_use = [gate.split("/")[-1] for gate in gates_to_use]
    score_data = score_data[score_data["gate"].isin(gates_to_use)]
    return score_data

def _read_technical_data(dataset_name: str):
    technical_data = pd.read_csv(os.path.join(DATASET_DIR,
                                          f"{dataset_name}/algorithm_comparison/Technicals.log"),
                                 index_col = False)
    technical_data["mem_used"] = technical_data["max_mem"] - technical_data["min_mem"]
    return technical_data


def _find_relevant_cluster(dataset: AnnData,
                           gate: str,
                           population_to_show: str,
                           algorithm: str) -> int:
    cluster_key = f"{population_to_show}_transformed_{algorithm}"
    cluster_counts = dataset.obs.groupby(gate).value_counts([cluster_key]).to_frame().reset_index()
    positives = cluster_counts[cluster_counts[gate] == gate]
    return positives.loc[positives["count"] == positives["count"].max(), cluster_key].iloc[0]

def _generate_figure_a(fig: Figure,
                       ax: Axes,
                       gs: GridSpec,
                       subfigure_label: str,
                       dataset: AnnData,
                       population_to_show: str,
                       algorithm: str,
                       graphical_abstract_markers: list[str],
                       graphical_abstract_gate: str):
    
    data = dataset.to_df(layer = "transformed")
    data["clus"] = dataset.obs[f"{population_to_show}_transformed_{algorithm}"]
    ax.axis("off")
    _figure_label(ax, subfigure_label, x = -0.24)
    fig_sgs = gs.subgridspec(2,2)

    empty_umap = fig.add_subplot(fig_sgs[0,0])

    _scanpy_vector_friendly()
    empty_umap_plot: Axes = fp.pl.umap(
        dataset,
        color = None,
        legend_loc = None,
        show = False,
        ax = empty_umap,
        s = 2
    )
    empty_umap_plot.set_title("Ungated cells", fontsize = TITLE_SIZE)
    empty_umap_plot.set_xlabel(empty_umap_plot.get_xlabel(), fontsize = AXIS_LABEL_SIZE)
    empty_umap_plot.set_ylabel(empty_umap_plot.get_ylabel(), fontsize = AXIS_LABEL_SIZE)
    empty_umap = fig.add_subplot(empty_umap_plot)

    cluster = fig.add_subplot(fig_sgs[0,1])
    cluster_plot: Axes = fp.pl.umap(
        dataset,
        color = f"{population_to_show}_transformed_{algorithm}",
        legend_loc = None,
        show = False,
        ax = cluster,
        s = 2
    )
    cluster_plot.set_title("clustering", fontsize = TITLE_SIZE)
    cluster_plot.set_xlabel(cluster_plot.get_xlabel(), fontsize = AXIS_LABEL_SIZE)
    cluster_plot.set_ylabel(cluster_plot.get_ylabel(), fontsize = AXIS_LABEL_SIZE)
    sns.reset_defaults()

    histogram = fig.add_subplot(fig_sgs[1,0])
    clus = _find_relevant_cluster(dataset,
                                  graphical_abstract_gate,
                                  population_to_show,
                                  algorithm)
    for marker in graphical_abstract_markers:
        sns.kdeplot(data = data[data["clus"] == clus],
                    x = marker,
                    ax = histogram,
                    label = marker)
    histogram.legend(**CENTERED_LEGEND_PARAMS)
    histogram.axvline(x = np.arcsinh(1),
                      color = "black",
                      linewidth = 1)
    histogram.text(x = 1,
                   y = histogram.get_ylim()[1]*1.2,
                   s = "Pos. Cutoff",
                   fontsize = AXIS_LABEL_SIZE)
    histogram.set_xlim(-1.5, 4.9)
    histogram.set_ylim(histogram.get_ylim()[0], histogram.get_ylim()[1]*1.5)

    histogram.set_title("per cluster:", fontsize = 8)
    histogram.set_xlabel("transformed expression", fontsize = AXIS_LABEL_SIZE)
    histogram.set_ylabel("Density", fontsize = AXIS_LABEL_SIZE)

    histogram.tick_params(axis='both', labelsize = AXIS_LABEL_SIZE)

    cell_type_annotation = fig.add_subplot(fig_sgs[1,1])

    _scanpy_vector_friendly()
    cell_type_annotation_plot: Axes = fp.pl.umap(
        dataset,
        color = graphical_abstract_gate,
        show = False,
        ax = cell_type_annotation,
        s = 2,
        cmap = "Set1"
    )
    handles, labels = cell_type_annotation.get_legend_handles_labels()
    cell_type_annotation.legend(handles, labels,
                                **CENTERED_LEGEND_PARAMS)
    cell_type_annotation_plot.set_title("annotation", fontsize = TITLE_SIZE)
    cell_type_annotation_plot.set_xlabel(cluster_plot.get_xlabel(), fontsize = AXIS_LABEL_SIZE)
    cell_type_annotation_plot.set_ylabel(cluster_plot.get_ylabel(), fontsize = AXIS_LABEL_SIZE)
    cell_type_annotation = fig.add_subplot(cell_type_annotation_plot)
    sns.reset_defaults()
    
    con1 = ConnectionPatch(xyA=(empty_umap_plot.get_xlim()[1],
                            empty_umap_plot.get_ylim()[0]), coordsA=empty_umap.transData, 
                       xyB=(cluster_plot.get_xlim()[0],
                            cluster_plot.get_ylim()[0]), coordsB=cluster.transData,
                       color = 'black',
                       arrowstyle = "->")
    fig.add_artist(con1)

    con2 = ConnectionPatch(xyA=(empty_umap_plot.get_xlim()[0],
                                empty_umap_plot.get_ylim()[0]), coordsA=cluster.transData, 
                           xyB=(histogram.get_xlim()[1],
                                histogram.get_ylim()[1]), coordsB=histogram.transData,
                           color = 'black',
                           arrowstyle = "->")
    fig.add_artist(con2)

    con3 = ConnectionPatch(xyA=(histogram.get_xlim()[1],
                                histogram.get_ylim()[1]), coordsA=histogram.transData, 
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
                       score_data: pd.DataFrame
                       ) -> None:
    
    ax.axis("off")
    _figure_label(ax, subfigure_label, x = -0.31)
    fig_sgs = gs.subgridspec(1,1)

    axis = fig.add_subplot(fig_sgs[0])
    plot_kwargs = {
        "data": score_data,
        "x": "algorithm",
        "y": UNSUPERVISED_SCORE,
        "hue": "gate",
        "ax": axis
    }
    sns.stripplot(**plot_kwargs,
                  **STRIPPLOT_PARAMS)
    sns.boxplot(**plot_kwargs,
                **BOXPLOT_PARAMS)
    handles, _ = axis.get_legend_handles_labels()
    gates = [gate.split("/")[-1] for gate in score_data["gate"].unique()]
    gates = [gate.replace("_", " ") for gate in gates]
    axis.legend(handles, gates,
                **CENTERED_LEGEND_PARAMS)
    # ["leiden", "parc", "flowsom", "phenograph"]
    axis.set_xticklabels(axis.get_xticklabels(),
                         **XTICKLABEL_PARAMS)
    axis.tick_params(**TICKPARAMS_PARAMS)
    axis.set_title("Cluster Algorithm Comparison", fontsize = TITLE_SIZE)
    axis.set_xlabel("")
    axis.set_ylabel(SCORE_LABEL, fontsize = AXIS_LABEL_SIZE)
    axis.set_ylim(SCORING_YLIMS)
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

    axis.set_xticklabels(axis.get_xticklabels(),
                         **XTICKLABEL_PARAMS)
    axis.tick_params(**TICKPARAMS_PARAMS)
    axis.set_title("Classification time", fontsize = TITLE_SIZE)
    axis.set_xlabel("")
    axis.set_ylabel("classification time\nper sample [s]", fontsize = AXIS_LABEL_SIZE)
    axis = fig.add_subplot(axis)

    return

def _generate_figure_d(fig: Figure,
                       ax: Axes,
                       gs: GridSpec,
                       subfigure_label: str,
                       algorithm_data: pd.DataFrame) -> None:
    
    ax.axis("off")
    _figure_label(ax, subfigure_label, x = -0.5)
    fig_sgs = gs.subgridspec(1,1)
    
    axis = fig.add_subplot(fig_sgs[0])
    plot_kwargs = {
        "data": algorithm_data,
        "x": "gate",
        "y": UNSUPERVISED_SCORE,
        "ax": axis
    }
    sns.stripplot(**plot_kwargs,
                  **STRIPPLOT_PARAMS)
    sns.boxplot(**plot_kwargs,
                **BOXPLOT_PARAMS)
    ticklabels = axis.get_xticklabels()
    _extract_population_from_gates_ticklabels(ticklabels)
    _remove_underscores_from_gates_ticklabels(ticklabels)
    axis.set_xticklabels(ticklabels,
                         **XTICKLABEL_PARAMS)
    axis.tick_params(**TICKPARAMS_PARAMS)
    axis.set_title(f"{SCORE_LABEL} per gate", fontsize = TITLE_SIZE)
    axis.set_xlabel("")
    axis.set_ylabel(SCORE_LABEL, fontsize = AXIS_LABEL_SIZE)
    axis.set_ylim(SCORING_YLIMS)
    axis = fig.add_subplot(axis)

    return

def _generate_figure_e(fig: Figure,
                       ax: Axes,
                       gs: GridSpec,
                       subfigure_label: str,
                       dataset: AnnData,
                       umap_markers: list[str],
                       vmax_map: dict[str, float]) -> None:
    
    ax.axis("off")
    _figure_label(ax, subfigure_label, x = -0.2)
    fig_sgs = gs.subgridspec(3,3)
    
    col_coord = -1

    _scanpy_vector_friendly()
    for i, marker in enumerate(umap_markers):
        row_coord = i%3
        if row_coord == 0:
            col_coord += 1
        marker_umap = fig.add_subplot(fig_sgs[col_coord,row_coord])
        marker_umap_plot: Axes = fp.pl.umap(
            dataset,
            color = marker,
            vmin = 0,
            vmax = vmax_map[marker],
            ax = marker_umap,
            show = False,
            s = 2
        )
        marker_umap_plot.set_title(marker, fontsize = TITLE_SIZE)
        marker_umap_plot.set_xlabel(marker_umap_plot.get_xlabel(), fontsize = AXIS_LABEL_SIZE)
        marker_umap_plot.set_ylabel(marker_umap_plot.get_ylabel(), fontsize = AXIS_LABEL_SIZE)

        marker_umap_plot._children[0].colorbar.ax.set_ylabel("")
        marker_umap_plot._children[0].colorbar.ax.tick_params(**TICKPARAMS_PARAMS)
        marker_umap = fig.add_subplot(marker_umap_plot)
    sns.reset_defaults()
    return

def _generate_figure_f(fig: Figure,
                       ax: Axes,
                       gs: GridSpec,
                       subfigure_label: str,
                       dataset: AnnData,
                       dataset_name: str,
                       gates_to_use: list[str],
                       gate_map: dict[str, str]) -> None:
    ax.axis("off")
    _figure_label(ax, subfigure_label, x = -0.1)
    fig_sgs = gs.subgridspec(2,8)
    
    umap_kwargs = {
        "legend_loc": None,
        "legend_fontsize": UMAP_LABEL_SIZE,
        "cmap": "Set1",
        "s": 1,
        "legend_fontoutline": 1
    }

    #gate_map = {k: v 
    #            for k, v in gate_map.items()
    #            if _is_in_gates_to_use(gates_to_use, k)}

    ## keep the order of the supplied gates to use    
    if dataset_name == "human_t_cells":
        gate_map = {k: gate_map[k] for k in gates_to_use}
    else:
        gate_map = {_extract_population_from_gates(k): gate_map[_extract_population_from_gates(k)]
                    for k in gates_to_use}


    _scanpy_vector_friendly() 
    for i, (original_gate, unsupervised_gate) in enumerate(gate_map.items()):
        if dataset_name == "human_t_cells":
            original_gate = _extract_population_from_gates(original_gate)
            unsupervised_gate = _extract_population_from_gates(unsupervised_gate)
        if i > 7:
            print(f"Warning... too many gates supplied, skipping {original_gate}")
            break
        gating_umap = fig.add_subplot(fig_sgs[0,i])
        gating_umap_plot: Axes = fp.pl.umap(
            dataset,
            color = original_gate,
            ax = gating_umap,
            show = False,
            **umap_kwargs
        )
        gating_umap_plot.set_title(f"{original_gate.replace('_', ' ')}\nExpert", fontsize = TITLE_SIZE)
        gating_umap_plot.set_xlabel(gating_umap_plot.get_xlabel(), fontsize = AXIS_LABEL_SIZE)
        gating_umap_plot.set_ylabel(gating_umap_plot.get_ylabel(), fontsize = AXIS_LABEL_SIZE)
        gating_umap = fig.add_subplot(gating_umap_plot)

        gating_umap = fig.add_subplot(fig_sgs[1,i])
        gating_umap_plot = fp.pl.umap(
            dataset,
            color = unsupervised_gate,
            ax = gating_umap,
            show = False,
            **umap_kwargs
        )
        gating_umap_plot.set_title("FACSPy", fontsize = TITLE_SIZE)
        gating_umap_plot.set_xlabel(gating_umap_plot.get_xlabel(), fontsize = AXIS_LABEL_SIZE)
        gating_umap_plot.set_ylabel(gating_umap_plot.get_ylabel(), fontsize = AXIS_LABEL_SIZE)
        gating_umap = fig.add_subplot(gating_umap_plot)
    sns.reset_defaults()

    return

def _generate_visualization_dataset(dataset_name: str,
                                    gating_strategy: dict,
                                    algorithm: str,
                                    population_to_show: str,
                                    pca_kwargs: Optional[dict],
                                    neighbors_kwargs: Optional[dict],
                                    umap_kwargs: Optional[dict],
                                    clustering_kwargs: Optional[dict],
                                    palette: str) -> AnnData:
    if pca_kwargs is None:
        pca_kwargs = {}
    if neighbors_kwargs is None:
        neighbors_kwargs = {}
    if umap_kwargs is None:
        umap_kwargs = {}
    if clustering_kwargs is None:
        clustering_kwargs = {}

    output_dir = os.path.join(DATASET_DIR, f"{dataset_name}")
    gated_dataset = os.path.join(output_dir, "gated_dataset.h5ad")

    if os.path.isfile(gated_dataset):
        print("Loading gated dataset...")
        dataset = fp.read_dataset(os.path.dirname(gated_dataset), "gated_dataset")
    else:
        print("Creating gated dataset...")
        dataset: AnnData = _generate_dataset(dataset_name = dataset_name,
                                             DATASET_DIR = DATASET_DIR,
                                             RAW_INPUT_DIR = RAW_INPUT_DIR)

        clf = fp.ml.unsupervisedGating(dataset,
                                       gating_strategy = gating_strategy,
                                       layer = "transformed",
                                       clustering_algorithm = algorithm,
                                       sensitivity = 1)
 
        print("Running Classification...")
        clf.identify_populations(cluster_kwargs = {"resolution_parameter": 10})

        fp.save_dataset(dataset,
                        output_dir,
                        "gated_dataset")

    fp.subset_gate(dataset, population_to_show)

    vis = sc.pp.subsample(dataset, n_obs = 10_000, copy = True)

    fp.tl.pca(vis,
              **pca_kwargs)
    fp.tl.neighbors(vis,
                    **neighbors_kwargs)
    fp.tl.umap(vis,
               **umap_kwargs)
    if algorithm == "parc":
        fp.tl.parc(vis,
                   **clustering_kwargs)
    else:
        raise NotImplementedError("please choose parc :D")
    
    color_palette = [colors.to_hex(color)
                     for color in sns.color_palette(palette)[:2]]
    for gate in vis.uns["gating_cols"]:
        gate_name = gate.split("/")[-1]

        vis.uns[f"{gate_name}_colors"] = color_palette
        fp.convert_gate_to_obs(vis, gate_name)
        cats = vis.obs[gate_name].cat.categories
        if len(cats) > 1:
            cell_type = [cat for cat in cats if cat != "other"][0]
            vis.obs[gate_name] = vis.obs[gate_name].cat.set_categories([cell_type, "other"]) 

    fp.save_dataset(vis,
                    output_dir = os.path.join(DATASET_DIR, dataset_name),
                    file_name = "visualization_clustered",
                    overwrite = True)
    return vis
  
def generate_unsupervised_characterization(dataset_name: Literal[
                                            "mouse_lineages_bm",
                                            "mouse_lineages_spl",
                                            "mouse_lineages_pb",
                                            "HIMC",
                                            "ZPM",
                                            "human_t_cells",
                                            "OMIP"
                                            ],
                                            algorithm: str,
                                            population_to_show: str,
                                            gates_to_use: list[str],
                                            graphical_abstract_gate: str,
                                            graphical_abstract_markers: list[str],
                                            umap_markers: list[str],
                                            vmax_map: dict[str, float],
                                            pca_kwargs: Optional[dict] = None,
                                            neighbors_kwargs: Optional[dict] = None,
                                            umap_kwargs: Optional[dict] = None,
                                            clustering_kwargs: Optional[dict] = None,
                                            palette: str = "Set1",
                                            save: Optional[str] = None,
                                            show: bool = True
                                            ) -> None:
    fp.settings.default_gate = population_to_show
    fp.settings.default_layer = "transformed"
 
    gating_strategy, gate_map = _get_gating_information(dataset_name)
    
    visualization_file = os.path.join(DATASET_DIR, f"{dataset_name}/visualization_clustered.h5ad")
    if os.path.isfile(visualization_file):
        vis = fp.read_dataset(os.path.dirname(visualization_file), "visualization_clustered")
    else:
        vis = _generate_visualization_dataset(dataset_name = dataset_name,
                                              gating_strategy = gating_strategy,
                                              algorithm = algorithm,
                                              population_to_show = population_to_show,
                                              pca_kwargs = pca_kwargs,
                                              neighbors_kwargs = neighbors_kwargs,
                                              umap_kwargs = umap_kwargs,
                                              clustering_kwargs = clustering_kwargs,
                                              palette = palette)

    score_data = _read_score_data(dataset_name = dataset_name,
                                  gates_to_use = gates_to_use)
    algorithm_data = score_data[score_data["algorithm"] == algorithm]
    
    technical_data = _read_technical_data(dataset_name = dataset_name)

    fig = plt.figure(layout = "constrained",
                     figsize = (FIGURE_WIDTH_FULL, FIGURE_HEIGHT_FULL))
    gs = GridSpec(ncols = 6,
                  nrows = 4,
                  figure = fig,
                  height_ratios = [0.8, 0.8, 0.8, 0.75])
    
    a_coords = gs[0, :3]
    b_coords = gs[0, 3:]
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

    _generate_figure_a(fig = fig,
                       ax = fig_a,
                       gs = a_coords,
                       subfigure_label = "A",
                       dataset = vis,
                       population_to_show = population_to_show,
                       algorithm = algorithm,
                       graphical_abstract_markers = graphical_abstract_markers,
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
                       algorithm_data = algorithm_data)
    
    _generate_figure_e(fig = fig,
                       ax = fig_e,
                       gs = e_coords,
                       subfigure_label = "E",
                       dataset = vis,
                       umap_markers = umap_markers,
                       vmax_map = vmax_map)
    
    _generate_figure_f(fig = fig,
                       ax = fig_f,
                       gs = f_coords,
                       subfigure_label = "F",
                       dataset_name = dataset_name,
                       gates_to_use = gates_to_use,
                       dataset = vis,
                       gate_map = gate_map)

    if save:
        plt.savefig(os.path.join(os.getcwd(), save),
                    dpi = DPI,
                    bbox_inches = "tight")
        png_path = save.replace(".pdf", ".png")
        plt.savefig(os.path.join(os.getcwd(), png_path),
                    dpi = DPI,
                    bbox_inches = "tight")
        
    if show:
        plt.show()

    return
