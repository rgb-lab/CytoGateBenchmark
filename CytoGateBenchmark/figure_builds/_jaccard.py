import os
import FACSPy as fp
from anndata import AnnData
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib import colors
import scanpy as sc

from pdf2image import convert_from_path

from PIL import Image, ImageChops

from sklearn.metrics import jaccard_score


from typing import Literal, Optional

from ._utils import (_figure_label,
                     _get_gating_information,
                     _scanpy_vector_friendly,
                     _trim,
                     STRIPPLOT_PARAMS,
                     BOXPLOT_PARAMS,
                     XTICKLABEL_PARAMS,
                     TICKPARAMS_PARAMS,
                     SCORING_YLIMS,
                     FIGURE_HEIGHT_FULL,
                     FIGURE_WIDTH_FULL,
                     UNSUPERVISED_SCORE,
                     TITLE_SIZE,
                     UMAP_LABEL_SIZE,
                     AXIS_LABEL_SIZE,
                     DPI)

RAW_INPUT_DIR = os.path.join(os.getcwd(), "datasets/")
DATASET_DIR = os.path.join(os.getcwd(), "figure_data/jaccard")

SCORE_LABEL = UNSUPERVISED_SCORE.replace("_", " ")

def _calculate_jaccard_scores(gating: pd.DataFrame,
                              cell_type: str) -> pd.DataFrame:
    populations = [f"{cell_type}_{i}" for i in range(1,6)]
    jac_df = pd.DataFrame()
    sids = gating["sample_ID"].unique()
    for population in populations:
        tmp_df = pd.DataFrame(data = {"population": [population for _ in range(len(sids))],
                                      "sample_ID": sids})
        jac_df = pd.concat([jac_df, tmp_df], axis = 0)

    for population in populations:
        for comp in populations:
            for sid in sids:
                sid_spec = gating[gating["sample_ID"] == sid]
                pop1 = sid_spec[population]
                pop2 = sid_spec[comp]
                jaccard = jaccard_score(pop1, pop2)
                jac_df.loc[(jac_df["sample_ID"] == sid) &
                           (jac_df["population"] == population), comp] = jaccard
    
    return pd.melt(jac_df,
                   id_vars=['population'],
                   value_vars=populations,
                   ignore_index=False,
                   var_name='comparison',
                   value_name='jaccard_score')


def _generate_figure_a(fig: Figure,
                       ax: Axes,
                       gs: GridSpec,
                       subfigure_label: str,
                       input_directory: str):
    
    ax.axis("off")
    _figure_label(ax, subfigure_label, x = -0.5)
    fig_sgs = gs.subgridspec(1,1)

    pil_image_lst = convert_from_path(os.path.join(input_directory, "full_gating_strategy.pdf"), dpi = DPI)
    pil_image = pil_image_lst[0]
    im = pil_image
    im = _trim(im)

    axis = fig.add_subplot(fig_sgs[0])
    axis.imshow(im)
    axis.axis("off")
    axis = fig.add_subplot(axis)
    return


def _generate_figure_b(fig: Figure,
                       ax: Axes,
                       gs: GridSpec,
                       subfigure_label: str,
                       dataset: AnnData,
                       cell_type: str,
                       input_directory: str,
                       jaccard_scores: pd.DataFrame,
                       population_to_show: Optional[str]) -> None:
    
    ax.axis("off")
    _figure_label(ax, subfigure_label, x = -0.5)
    fig_sgs = gs.subgridspec(12,5)

    for i in range(1,6):
        if i != 5:
            if population_to_show:
                path = os.path.join(input_directory, f"{population_to_show}_{i}.pdf")
            else:
                path = os.path.join(input_directory, f"{cell_type}_{i}.pdf")
            pop1 = convert_from_path(path, dpi = DPI)[0]
            pop1 = _trim(pop1)
            pop1_plot = fig.add_subplot(fig_sgs[0:7, i-1])
            pop1_plot.imshow(pop1)
            pop1_plot.axis("off")
            pop1_plot = fig.add_subplot(pop1_plot)

        _scanpy_vector_friendly()
        pop1_umap = fig.add_subplot(fig_sgs[7:10, i-1])
        pop1_umap_plot: Axes = fp.pl.umap(dataset,
                                          color = f"{cell_type}_{i}",
                                          show = False,
                                          ax = pop1_umap,
                                          legend_loc = None,
                                          legend_fontsize = UMAP_LABEL_SIZE)
        pop1_umap_plot.set_xlabel(pop1_umap_plot.get_xlabel(), fontsize = AXIS_LABEL_SIZE)
        pop1_umap_plot.set_ylabel(pop1_umap_plot.get_ylabel(), fontsize = AXIS_LABEL_SIZE)
        pop1_umap_plot.set_title(f"{cell_type}_{i}", fontsize = TITLE_SIZE)
        pop1_umap = fig.add_subplot(pop1_umap_plot)
        sns.reset_defaults()

        jac_score = fig.add_subplot(fig_sgs[10:12, i-1])
        jaccard_data = jaccard_scores[jaccard_scores["population"] == f"{cell_type}_{i}"]
        plot_kwargs = {
            "data": jaccard_data,
            "x": "comparison",
            "y": "jaccard_score",
            "ax": jac_score,
        }
        sns.stripplot(**plot_kwargs,
                      **STRIPPLOT_PARAMS)
        sns.boxplot(**plot_kwargs,
                    **BOXPLOT_PARAMS)
        jac_score.tick_params(**TICKPARAMS_PARAMS)
        jac_score.set_xticklabels(jac_score.get_xticklabels(),
                                  **XTICKLABEL_PARAMS)
        jac_score.set_xlabel("")
        jac_score.set_ylabel(SCORE_LABEL, fontsize = 6)
        jac_score.set_ylim(SCORING_YLIMS)
        jac_score = fig.add_subplot(jac_score)

    return

def _generate_figure_c(fig: Figure,
                       ax: Axes,
                       gs: GridSpec,
                       subfigure_label: str,
                       dataset: AnnData,
                       umap_markers: list[str],
                       vmax_map: dict[str, float]) -> None:
    
    ax.axis("off")
    _figure_label(ax, subfigure_label, x = -0.5)
    fig_sgs = gs.subgridspec(2,4)
    
    col_coord = -1
    _scanpy_vector_friendly()
    for i, marker in enumerate(umap_markers[:8]):
        row_coord = i%4
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
        marker_umap_plot._children[0].colorbar.ax.tick_params(labelsize = AXIS_LABEL_SIZE)
        marker_umap = fig.add_subplot(marker_umap_plot)
    sns.reset_defaults()
    return

def _create_dataset(dataset_name,
                    input_directory: str,
                    organ: Optional[str]) -> AnnData:
    if "mouse_lineage" in dataset_name:
        workspace = fp.dt.FlowJoWorkspace(file = os.path.join(input_directory, f"jaccard_gates_{organ}.wsp"))
        metadata = fp.dt.Metadata(file = os.path.join(input_directory, f"metadata_{organ}.csv"))
        panel = fp.dt.Panel(file = os.path.join(input_directory, "panel.csv"))
        cofactors = fp.dt.CofactorTable(file = os.path.join(input_directory, f"cofactors_{organ}.csv"))
    else:
        workspace = fp.dt.FlowJoWorkspace(file = os.path.join(input_directory, f"jaccard_gates.wsp"))
        metadata = fp.dt.Metadata(file = os.path.join(input_directory, f"metadata.csv"))
        panel = fp.dt.Panel(file = os.path.join(input_directory, "panel.csv"))
        cofactors = fp.dt.CofactorTable(file = os.path.join(input_directory, f"cofactors.csv"))

    dataset = fp.create_dataset(input_directory = input_directory,
                                metadata = metadata,
                                panel = panel,
                                workspace = workspace,
                                )
    dataset = dataset[dataset.obs["staining"] == "stained",:].copy()
    dataset = fp.transform(dataset,
                           transform = "asinh",
                           cofactor_table = cofactors,
                           key_added = "transformed",
                           copy = True)
    return dataset

def _generate_gated_dataset(dataset_name: str,
                            input_directory: str,
                            palette: str,
                            cell_type: str,
                            dimred_params: dict,
                            gating_strategy: dict,
                            population_to_show: Optional[str]) -> tuple[AnnData, pd.DataFrame]:

    output_dir = os.path.join(DATASET_DIR, f"{dataset_name}")
    gated_dataset = os.path.join(output_dir, "gated_dataset.h5ad")

    if os.path.isfile(gated_dataset):
        print("Loading gated dataset...")
        dataset = fp.read_dataset(os.path.dirname(gated_dataset), "gated_dataset")
    else:
        print("Creating gated dataset...")
        dataset = _create_dataset(dataset_name = dataset_name,
                                  input_directory = input_directory,
                                  organ = dataset_name.split("_")[2] if "mouse" in dataset_name else None)
    
        clf = fp.ml.unsupervisedGating(dataset,
                                       gating_strategy = gating_strategy,
                                       layer = "transformed",
                                       clustering_algorithm = "parc",
                                       intervals = [0.6, 0.8])
        clf.identify_populations(cluster_kwargs = {"resolution_parameter": 10})

        fp.save_dataset(dataset,
                        output_dir,
                        "gated_dataset")
    
    color_palette = [colors.to_hex(color)
                     for color in sns.color_palette(palette)[:2]]

    for gate in dataset.uns["gating_cols"]:
        gate_name = gate.split("/")[-1]
        dataset.uns[f"{gate_name}_colors"] = color_palette
        fp.convert_gate_to_obs(dataset, gate_name)
        cats = dataset.obs[gate_name].cat.categories
        if len(cats)>1:
            _cell_type = [cat for cat in cats if cat != "other"][0]
            dataset.obs[gate_name] = dataset.obs[gate_name].cat.set_categories([_cell_type, "other"])
        

    dataset.obs[f"{cell_type}_5"] = dataset.obs[f"unsup_{cell_type}"]
    dataset.obs[f"{cell_type}_5"] = dataset.obs[f"{cell_type}_5"].map({f"unsup_{cell_type}": f"{cell_type}_5",
                                                                       "other": "other"})
    dataset.uns[f"{cell_type}_5_colors"] = color_palette

    gating = pd.DataFrame(data = dataset.obsm["gating"].toarray(),
                          index = dataset.obs_names,
                          columns = [gate.split("/")[-1] for gate in dataset.uns["gating_cols"]])
    gating[f"{cell_type}_5"] = gating[f"unsup_{cell_type}"]
    gating["sample_ID"] = dataset.obs["sample_ID"].tolist()
    gating.to_csv(os.path.join(output_dir, f"gating_{cell_type}.csv"))

    if population_to_show:
        fp.convert_gate_to_obs(dataset, population_to_show)
        cell_subset = dataset[dataset.obs[population_to_show] == population_to_show].copy()

    else:
        cell_subset = dataset[
            (dataset.obs[f"{cell_type}_1"] == f"{cell_type}_1") |
            (dataset.obs[f"{cell_type}_2"] == f"{cell_type}_2") |
            (dataset.obs[f"{cell_type}_3"] == f"{cell_type}_3") |
            (dataset.obs[f"{cell_type}_4"] == f"{cell_type}_4") |
            (dataset.obs[f"{cell_type}_5"] == f"{cell_type}_5")
        ].copy()

    fp.settings.default_gate = fp._utils.find_parent_population(
        fp._utils.find_gate_path_of_gate(cell_subset, cell_type)
    )

    print("Running Dimensionality Reduction...")
    fp.tl.pca(cell_subset, **dimred_params)
    fp.tl.neighbors(cell_subset, **dimred_params)
    fp.tl.umap(cell_subset, **dimred_params)

    fp.save_dataset(cell_subset,
                    output_dir,
                    cell_type)

    return cell_subset, gating


def generate_jaccard_comparison(dataset_name: Literal[
                                    "mouse_lineages_bm",
                                    "mouse_lineages_spl",
                                    "mouse_lineages_pb",
                                    "HIMC",
                                    "ZPM",
                                    "human_t_cells",
                                    "OMIP"
                                    ],
                                flowjo_plot_directory: str,
                                cell_type: str,
                                dimred_params: Optional[dict],
                                umap_markers: list[str],
                                vmax_map: dict,
                                base_population: str,
                                population_to_show: Optional[str] = None,
                                save: Optional[str] = None,
                                show: bool = True,
                                gating_strategy: Optional[dict] = None,
                                palette: str = "Set1",
                                ):
    fp.settings.default_layer = "transformed"
    fp.settings.default_gate = base_population
    
    if not dimred_params:
        dimred_params = {}
    if not gating_strategy:
        gating_strategy, _ = _get_gating_information(dataset_name)

    gated_dataset = os.path.join(DATASET_DIR, f"{dataset_name}/{cell_type}.h5ad")
    if os.path.isfile(gated_dataset):
        print("Loading cell type specific dataset")
        dataset = fp.read_dataset(os.path.dirname(gated_dataset), cell_type)
        gating = pd.read_csv(os.path.join(os.path.dirname(gated_dataset), f"gating_{cell_type}.csv"))
    else:
        print("Creating dataset...")
        if "mouse" in dataset_name:
            input_dir = os.path.join(RAW_INPUT_DIR, f"{'_'.join(dataset_name.split('_')[:2])}")
        else:
            input_dir = os.path.join(RAW_INPUT_DIR, f"{dataset_name}")
        dataset, gating = _generate_gated_dataset(dataset_name = dataset_name,
                                                  input_directory = input_dir,
                                                  palette = palette,
                                                  cell_type = cell_type,
                                                  dimred_params = dimred_params,
                                                  gating_strategy = gating_strategy,
                                                  population_to_show = population_to_show)

    jaccard_score_file = os.path.join(DATASET_DIR, f"{dataset_name}/{cell_type}_jaccards.csv")
    if os.path.isfile(jaccard_score_file):
        print("Loading Jaccard Scores...")
        jaccard_scores = pd.read_csv(jaccard_score_file)
    else:
        print("Calculating Jaccard Scores...")
        jaccard_scores = _calculate_jaccard_scores(gating, cell_type)
        jaccard_scores.to_csv(jaccard_score_file)

    fig = plt.figure(layout = "constrained",
                     figsize = (FIGURE_WIDTH_FULL, FIGURE_HEIGHT_FULL))
    gs = GridSpec(ncols = 1,
                  nrows = 3,
                  figure = fig,
                  height_ratios = [0.5,1.4,0.9])

    
    a_coords = gs[0, :]
    b_coords = gs[1, :]
    c_coords = gs[2, :]

    fig_a = fig.add_subplot(a_coords)
    fig_b = fig.add_subplot(b_coords)
    fig_c = fig.add_subplot(c_coords)

    _generate_figure_a(fig = fig,
                       ax = fig_a,
                       gs = a_coords,
                       subfigure_label = "A",
                       input_directory = flowjo_plot_directory 
                       )

    _generate_figure_b(fig = fig,
                       ax = fig_b,
                       gs = b_coords,
                       subfigure_label = "B",
                       dataset = dataset,
                       input_directory = flowjo_plot_directory,
                       jaccard_scores = jaccard_scores,
                       cell_type = cell_type,
                       population_to_show = population_to_show)

    _generate_figure_c(fig = fig,
                       ax = fig_c,
                       gs = c_coords,
                       subfigure_label = "C",
                       dataset = dataset,
                       vmax_map = vmax_map,
                       umap_markers = umap_markers)

    if save:
        plt.savefig(os.path.join(os.getcwd(), save),
                    dpi = DPI,
                    bbox_inches = "tight")
        
    if show:
        plt.show()



    return
