import os

from matplotlib import pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import matplotlib.patches as mpatches

from matplotlib.colors import LogNorm, SymLogNorm
import seaborn as sns

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from pdf2image import convert_from_path
from ..figure_builds._supervised_characterization_overview import _read_all_scores
from ..figure_builds._utils import (_figure_label,
                                    _trim,
                                    FIGURE_WIDTH_FULL,
                                    FIGURE_HEIGHT_HALF,
                                    DPI)

RAW_INPUT_DIR = os.path.join(os.getcwd(), "datasets/")
IMG_DIR = os.path.join(os.getcwd(), "images/")
DATA_DIR = os.path.join(os.getcwd(), "data/")

DATASET_MAP = {
    "Mouse Lineages BM": "Flow Cytometry: mouse lineages bone marrow (dataset 1)",
    "Mouse Lineages PB": "Flow Cytometry: mouse lineages peripheral blood (dataset 2)",
    "Mouse Lineages SPL": "Flow Cytometry: mouse lineages spleen (dataset 3)",
    "HIMC": "Mass Cytometry: human Leukocytes (dataset 7)",
    "Human T-Cells": "Flow Cytometry: human T cells (dataset 4)",
    "ZPM": "Flow Cytometry: human T cells (dataset 5)",
    "OMIP": "Spectral Flow Cytometry: human leukocytes (dataset 6)"
}

def _generate_subfigure_c(fig: Figure,
                          ax: Axes,
                          gs: GridSpec,
                          subfigure_label: str):
    ax.axis("off")
    _figure_label(ax, subfigure_label, x = -0.12)
    axes = gs.subgridspec(2,3, wspace = 0.3, hspace = 0.4)

    data = pd.read_csv(os.path.join(DATA_DIR, "Figure2_hyperparameter_tuning_score_data.csv"), index_col = False)
    data = data[data["sampling"] == False]
    data = data[data["tuned"] == True]
    rev_map = {value: key for key, value in DATASET_MAP.items()}
    data["dataset"] = data["dataset"].map(rev_map)
    df = data.groupby(["dataset", "train_size", "algorithm", "tuned"]).median("f1_score").reset_index()
    df = df.pivot(index='dataset', columns=["algorithm", "train_size", "tuned"], values='f1_score')
    df = df.T.reset_index().fillna(0)
    data = df
    # Melt the dataframe to long format for easier plotting
    data_melted = pd.melt(data, id_vars=['algorithm', 'train_size', 'tuned'], 
                          value_vars=['HIMC', 'Human T-Cells', 'Mouse Lineages BM', 'Mouse Lineages PB', 'Mouse Lineages SPL', 'OMIP'],
                          var_name='dataset', value_name='f1_score')
    
    # Function to process algorithm names
    def process_algorithm_name(name):
        if name.endswith("Classifier"):
            return name.replace("Classifier", "").strip()
        return name
    
    # Apply the function to the algorithm column
    data_melted['algorithm'] = data_melted['algorithm'].apply(process_algorithm_name)
    
    # Create the smaller plot excluding the ZPM dataset
    # fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(4,2.6), sharey=False, sharex=True)

    # axes = axes.flatten()
    
    # List of datasets
    datasets_mouse = [dataset for dataset in data_melted['dataset'].unique() if "Mouse Lineages" in dataset]
    datasets_other = [dataset for dataset in data_melted['dataset'].unique() if "Mouse Lineages" not in dataset]
    
    # Plotting datasets with "Mouse Lineages" in the top row
    for i, dataset in enumerate(datasets_mouse):
        subset = data_melted[data_melted['dataset'] == dataset]
        axis = fig.add_subplot(axes[0,i])
        plot = sns.lineplot(
            data=subset, 
            x='train_size', y='f1_score', 
            hue='algorithm', marker='o', markersize=0, linewidth=0.8,
            ax=axis, palette='tab10', legend=True, ci = False
        )
        plot.legend().remove()
        plot.set_title(dataset, fontsize=7, pad=3)
        plot.set_xscale('log')
        plot.set_xlabel('')
        plot.set_ylabel('')
        plot.set_xticklabels([])
        plot.tick_params(axis='both', which='major', labelsize=6, pad=1)
    
    # Plotting other datasets in the second row
    for i, dataset in enumerate(datasets_other):
        subset = data_melted[data_melted['dataset'] == dataset]
        axis = fig.add_subplot(axes[1,i])
        plot = sns.lineplot(
            data=subset, 
            x='train_size', y='f1_score', 
            hue='algorithm', marker='o', markersize=0, linewidth=0.8,
            ax=axis, palette='tab10', legend=True, ci = False
        )
        handles, labels = plot.get_legend_handles_labels()
        plot.legend().remove()
        plot.set_title(dataset, fontsize=7, pad=3)
        plot.set_xscale('log')
        plot.set_xlabel('')
        plot.set_ylabel('')
        plot.tick_params(axis='both', which='major', labelsize=6, pad=1)
    
    # Set only one label per axis, centered
    ax.text(0.5, -0.2, 'Training Size', ha='center', fontsize=8)
    ax.text(-0.11, 0.5, 'F1 Score', va='center', rotation='vertical', fontsize=8)
    
    # Create a single legend with shortened line labels
    ax.legend(handles, labels,
                  loc='upper center',
                  ncol = len(data_melted["dataset"].unique())/2,
                  bbox_to_anchor=(0.5, -0.2),
                  title='Algorithm', fontsize=6, title_fontsize=6, handlelength=1, handleheight=0.8)

    
    # Adjust layout and show the plot
    # plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust rect to make room for the legend
    # plt.show()
    # plt.tight_layout(h_pad = 2, w_pad = 2)

def _generate_subfigure_b(fig: Figure,
                          ax: Axes,
                          gs: GridSpec,
                          subfigure_label: str):
    ax.axis("off")
    _figure_label(ax, subfigure_label, x = -1)
    fig_sgs = gs.subgridspec(1,1)

    scores = pd.read_csv(os.path.join(DATA_DIR, "Figure2_algcomp_score_data.csv"), index_col = False)
    scores = scores[scores["sampling"] == False]
    rev_map = {value: key for key, value in DATASET_MAP.items()}
    scores["dataset"] = scores["dataset"].map(rev_map)
    scores = scores.groupby(["dataset", "algorithm", "sample_ID"]).mean("f1_score")

    data = scores
    # Calculate the mean f1_scores for each algorithm-dataset combination
    mean_f1_scores = data.groupby(['dataset', 'algorithm'])['f1_score'].median().unstack().fillna(0)
    
    # Calculate the ranking for each algorithm within each dataset
    ranking_f1_scores = mean_f1_scores.rank(axis=1, ascending=False)
    
    # Calculate the mean rank for each algorithm across datasets
    mean_rank_per_algorithm = ranking_f1_scores.mean().sort_values()
    
    # Reorder the columns of mean_f1_scores and ranking_f1_scores based on the mean rank
    mean_f1_scores_ordered = mean_f1_scores[mean_rank_per_algorithm.index]
    ranking_f1_scores_ordered = ranking_f1_scores[mean_rank_per_algorithm.index]
    
    # Transpose the mean and ranking dataframes to invert the axes
    mean_f1_scores_ordered_transposed = mean_f1_scores_ordered.T
    ranking_f1_scores_ordered_transposed = ranking_f1_scores_ordered.T


    # Sort the transposed dataframes to have the best ranking classifiers at the top
    mean_rank_per_algorithm_transposed = ranking_f1_scores_ordered_transposed.mean(axis=1).sort_values(ascending=False)
    mean_f1_scores_ordered_transposed_sorted = mean_f1_scores_ordered_transposed.loc[mean_rank_per_algorithm_transposed.index]
    ranking_f1_scores_ordered_transposed_sorted = ranking_f1_scores_ordered_transposed.loc[mean_rank_per_algorithm_transposed.index]
    
    # Normalize the mean f1_score for dot sizes for a smaller figure to prevent overlap
    dot_size_f1_rescaled_small_sorted = (mean_f1_scores_ordered_transposed_sorted - mean_f1_scores_ordered_transposed_sorted.min().min()) / (mean_f1_scores_ordered_transposed_sorted.max().max() - mean_f1_scores_ordered_transposed_sorted.min().min()) * 80 + 20
    
    # Define new unique sizes and labels for the f1 score legend to range from 0 to 1 and avoid overlap
    unique_sizes_rescaled = [20, 40, 60, 80, 100]  # Adjust sizes further to avoid overlap
    size_labels_rescaled = [f'{size / 100:.2f}' for size in unique_sizes_rescaled]  # Labels ranging from 0 to 1
    
    # Create the dot-heatmap using transposed and sorted data with size representing rescaled f1-score and color representing ranking with RdYlBu colormap
    ax = fig.add_subplot(fig_sgs[0])
    # fig, ax = plt.subplots(figsize=(3, 5))
    
    # Plot each point as a dot with marker value set to 'o' and edgecolor 'black'
    for x in range(mean_f1_scores_ordered_transposed_sorted.shape[1]):
        for y in range(mean_f1_scores_ordered_transposed_sorted.shape[0]):
            mean_score = mean_f1_scores_ordered_transposed_sorted.iloc[y, x]
            size = dot_size_f1_rescaled_small_sorted.iloc[y, x]
            rank = ranking_f1_scores_ordered_transposed_sorted.iloc[y, x]
            color = plt.cm.RdYlBu(rank / ranking_f1_scores_ordered_transposed_sorted.max().max())
            ax.scatter(x, y, s=size, color=color, marker='o', edgecolor='black')
    
    # Customize the plot without the internal grid lines
    ax.set_xticks(np.arange(mean_f1_scores_ordered_transposed_sorted.shape[1]))
    ax.set_xticklabels(mean_f1_scores_ordered_transposed_sorted.columns,
                       rotation=45, horizontalalignment='right', fontsize=6)
    ax.set_yticks(np.arange(mean_f1_scores_ordered_transposed_sorted.shape[0]))
    ax.set_yticklabels(mean_f1_scores_ordered_transposed_sorted.index,
                       fontsize=6,
                       rotation=45)
    ax.set_xlim(-0.5, mean_f1_scores_ordered_transposed_sorted.shape[1] - 0.5)
    ax.set_ylim(-0.5, mean_f1_scores_ordered_transposed_sorted.shape[0] - 0.5)
    
    # Remove the grid lines
    ax.grid(False)


    # Create legend patches for ranking
    unique_ranks = range(1, int(ranking_f1_scores_ordered.max().max()) + 1)
    colors = [plt.cm.RdYlBu(rank / ranking_f1_scores_ordered.max().max()) for rank in unique_ranks]
    patches = [mpatches.Patch(color=colors[i], label=i+1) for i in range(len(unique_ranks))]

    # Adjusted positions for the legends
    rank_legend_bbox = (1.1, 1)
    size_legend_bbox = (1.04, 0.3)
    
    # Add the customized legend for ranking
    rank_legend = ax.legend(handles=patches,
                            title='',
                            loc='upper left',
                            bbox_to_anchor=rank_legend_bbox,
                            frameon=False,
                            handletextpad=1,
                            handlelength=0.6,
                            fontsize = 6.5)

    ax.text(10, 10, 'classifier rank per dataset', fontsize=7, ha='center', va='center', rotation=-90)

    # Add the customized legend for f1 score size
    size_patches_rescaled = [plt.scatter([], [], s=size, color='gray', label=label, marker='o', edgecolor='black') for size, label in zip(unique_sizes_rescaled, size_labels_rescaled)]
    legend1 = ax.legend(handles=size_patches_rescaled,
                        title='F1 score',
                        loc='lower left',
                        bbox_to_anchor=(1.01, 0),
                        frameon=False,
                        scatterpoints=1,
                        fontsize = 7)
    legend1.get_title().set_fontsize(7)
    
    # Add back the first legend
    fig.add_artist(legend1)
    fig.add_artist(rank_legend)

    axis = fig.add_subplot(ax)
    

def _generate_subfigure_a(fig: Figure,
                          ax: Axes,
                          gs: GridSpec,
                          subfigure_label: str):
    ax.axis("off")
    _figure_label(ax, subfigure_label, x = -0.12)
    fig_sgs = gs.subgridspec(1,1)

    pil_image_lst = convert_from_path(os.path.join(IMG_DIR, 'Fig1_Graphical_abstract.pdf'), dpi = DPI)
    pil_image = pil_image_lst[0]
    im = pil_image
    im = _trim(im)

    axis = fig.add_subplot(fig_sgs[0])
    axis.imshow(im)
    axis.axis("off")
    axis = fig.add_subplot(axis)

def figure_1():

    fig = plt.figure(layout = "constrained",
                     figsize = (FIGURE_WIDTH_FULL, FIGURE_HEIGHT_HALF))

    gs = GridSpec(ncols = 5,
                  nrows = 3,
                  height_ratios = [1,1,1.5],
                  wspace = 0,
                  hspace = 0.3)

    a_coords = gs[0:2,0:3]
    b_coords = gs[:,4]
    c_coords = gs[2:,0:3]

    fig_a = fig.add_subplot(a_coords)
    fig_b = fig.add_subplot(b_coords)
    fig_c = fig.add_subplot(c_coords)

    _generate_subfigure_a(fig = fig,
                          ax = fig_a,
                          gs = a_coords,
                          subfigure_label = "A")

    _generate_subfigure_b(fig = fig,
                          ax = fig_b,
                          gs = b_coords,
                          subfigure_label = "B")

    _generate_subfigure_c(fig = fig,
                          ax = fig_c,
                          gs = c_coords,
                          subfigure_label = "C")

    plt.savefig("Figure1.pdf", dpi = 300, bbox_inches = "tight")
    plt.show()
