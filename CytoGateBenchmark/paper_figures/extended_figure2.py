from matplotlib import pyplot as plt

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
from matplotlib.patches import Rectangle

from pdf2image import convert_from_path
from ..figure_builds._supervised_characterization_overview import _read_all_scores
from ..figure_builds._utils import (_figure_label,
                                    _trim,
                                    FIGURE_WIDTH_FULL,
                                    FIGURE_HEIGHT_FULL,
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

def _generate_subfigure_a(fig: Figure,
                          ax: Axes,
                          gs: GridSpec,
                          subfigure_label: str):
    ax.axis("off")
    _figure_label(ax, subfigure_label, x = -1)
    fig_sgs = gs.subgridspec(1,1)

    tech_data = pd.read_csv(os.path.join(DATA_DIR, "Figure2_algcomp_technical_data.csv"), index_col = False)
    tech_data = tech_data[tech_data["sampling"] == False]
    rev_map = {value: key for key, value in DATASET_MAP.items()}
    tech_data["dataset"] = tech_data["dataset"].map(rev_map)
    tech_data = tech_data.groupby(["dataset", "algorithm", "sample_ID"]).mean("train_time")

    # Calculate the total time by adding "train_time" and "pred_time_val"
    tech_data['total_time'] = tech_data['train_time'] + tech_data['pred_time_val']
    
    # Calculate the mean total_time for each algorithm-dataset combination
    mean_total_time = tech_data.groupby(['dataset', 'algorithm'])['total_time'].mean().unstack().dropna()
    
    # Calculate the ranking for each algorithm within each dataset
    ranking_total_time = mean_total_time.rank(axis=1, ascending=True)
    
    # Calculate the mean rank for each algorithm across datasets
    mean_rank_per_algorithm_total = ranking_total_time.mean().sort_values()
    
    # Reorder the columns of mean_total_time and ranking_total_time based on the mean rank
    mean_total_time_ordered = mean_total_time[mean_rank_per_algorithm_total.index]
    ranking_total_time_ordered = ranking_total_time[mean_rank_per_algorithm_total.index]
    
    # Transpose the mean and ranking dataframes to invert the axes
    mean_total_time_ordered_transposed = mean_total_time_ordered.T
    ranking_total_time_ordered_transposed = ranking_total_time_ordered.T
    
    # Sort the transposed dataframes to have the best ranking classifiers at the top
    mean_rank_per_algorithm_total_transposed = ranking_total_time_ordered_transposed.mean(axis=1).sort_values(ascending=False)
    mean_total_time_ordered_transposed_sorted = mean_total_time_ordered_transposed.loc[mean_rank_per_algorithm_total_transposed.index]
    ranking_total_time_ordered_transposed_sorted = ranking_total_time_ordered_transposed.loc[mean_rank_per_algorithm_total_transposed.index]
    
    # Normalize the mean total_time for dot sizes for a smaller figure to prevent overlap (log scale)
    dot_size_total_time_rescaled_small_sorted = np.log(mean_total_time_ordered_transposed_sorted + 1)
    
    # Define new unique sizes and labels for the total_time legend to range from 0.5 to 4200 and avoid overlap
    legend_values = [0.5, 10, 75, 600, 4200]
    legend_sizes = [np.log(value + 1) * 15 + 5 for value in legend_values]  # Ensure same scaling
    
    # Create the dot-heatmap using transposed and sorted data with size representing rescaled total_time and color representing ranking with RdYlBu colormap
    ax = fig.add_subplot(fig_sgs[0])
    
    # Plot each point as a dot with marker value set to 'o', edgecolor 'black', and linewidth 0.2
    for x in range(mean_total_time_ordered_transposed_sorted.shape[1]):
        for y in range(mean_total_time_ordered_transposed_sorted.shape[0]):
            mean_score = mean_total_time_ordered_transposed_sorted.iloc[y, x]
            size = dot_size_total_time_rescaled_small_sorted.iloc[y, x] * 15 + 5  # Further rescaled and adjusted for log scale
            rank = ranking_total_time_ordered_transposed_sorted.iloc[y, x]
            color = plt.cm.RdYlBu(rank / ranking_total_time_ordered_transposed_sorted.max().max())
            ax.scatter(x, y, s=size, color=color, marker='o', edgecolor='black', linewidth=0.2)
    
    # Customize the plot without the internal grid lines
    ax.set_xticks(np.arange(mean_total_time_ordered_transposed_sorted.shape[1]))
    ax.set_xticklabels(mean_total_time_ordered_transposed_sorted.columns, rotation=45, horizontalalignment='right', fontsize=8)
    ax.set_yticks(np.arange(mean_total_time_ordered_transposed_sorted.shape[0]))
    ax.set_yticklabels(mean_total_time_ordered_transposed_sorted.index, fontsize=8, rotation=45)
    ax.set_xlim(-0.5, mean_total_time_ordered_transposed_sorted.shape[1] - 0.5)
    ax.set_ylim(-0.5, mean_total_time_ordered_transposed_sorted.shape[0] - 0.5)
    
    # Remove the grid lines
    ax.grid(False)
    
    # Add the customized legend for total_time size
    size_patches_rescaled = [plt.scatter([], [], s=size, color='gray', label=f'{value:.1f}', marker='o', edgecolor='black', linewidth=0.2) for size, value in zip(legend_sizes, legend_values)]
    legend1 = ax.legend(handles=size_patches_rescaled, title='total time [s]',
                        loc='upper left', bbox_to_anchor=(1.01, 0.3), frameon=False, scatterpoints=1,
                        fontsize = 8)
    legend1.get_title().set_fontsize(7)

    # Create legend patches for ranking
    unique_ranks = range(1, int(ranking_total_time_ordered.max().max()) + 1)
    colors = [plt.cm.RdYlBu(rank / ranking_total_time_ordered.max().max()) for rank in unique_ranks]
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

    fig.add_artist(rank_legend)
    fig.add_artist(legend1)
    axis = fig.add_subplot(ax)


def _generate_subfigure_b(fig: Figure,
                          ax: Axes,
                          gs: GridSpec,
                          subfigure_label: str):
    ax.axis("off")
    _figure_label(ax, subfigure_label, x = -1)
    fig_sgs = gs.subgridspec(1,1)

    tech_data = pd.read_csv(os.path.join(DATA_DIR, "Figure2_algcomp_technical_data.csv"), index_col = False)
    tech_data = tech_data[tech_data["sampling"] == False]
    rev_map = {value: key for key, value in DATASET_MAP.items()}
    tech_data["dataset"] = tech_data["dataset"].map(rev_map)

    # Calculate the mean mem_used for each algorithm-dataset combination
    mean_mem_used = tech_data.groupby(['dataset', 'algorithm'])['mem_used'].mean().unstack().dropna()
    
    # Calculate the ranking for each algorithm within each dataset
    ranking_mem_used = mean_mem_used.rank(axis=1, ascending=True)
    
    # Calculate the mean rank for each algorithm across datasets
    mean_rank_per_algorithm_mem = ranking_mem_used.mean().sort_values()
    
    # Reorder the columns of mean_mem_used and ranking_mem_used based on the mean rank
    mean_mem_used_ordered = mean_mem_used[mean_rank_per_algorithm_mem.index]
    ranking_mem_used_ordered = ranking_mem_used[mean_rank_per_algorithm_mem.index]
    
    # Transpose the mean and ranking dataframes to invert the axes
    mean_mem_used_ordered_transposed = mean_mem_used_ordered.T
    ranking_mem_used_ordered_transposed = ranking_mem_used_ordered.T
    
    # Sort the transposed dataframes to have the best ranking classifiers at the top
    mean_rank_per_algorithm_mem_transposed = ranking_mem_used_ordered_transposed.mean(axis=1).sort_values(ascending=False)
    mean_mem_used_ordered_transposed_sorted = mean_mem_used_ordered_transposed.loc[mean_rank_per_algorithm_mem_transposed.index]
    ranking_mem_used_ordered_transposed_sorted = ranking_mem_used_ordered_transposed.loc[mean_rank_per_algorithm_mem_transposed.index]
    
    # Normalize the mean mem_used for dot sizes for a smaller figure to prevent overlap (log scale)
    dot_size_mem_used_rescaled_small_sorted = np.log(mean_mem_used_ordered_transposed_sorted + 1)
    
    # Define new unique sizes and labels for the mem_used legend to avoid overlap
    legend_values = [0, 1, 10, 100, 1000]
    legend_sizes = [np.log(value + 1) * 15 + 5 if value != 0 else 5 for value in legend_values]  # Ensure same scaling and assign a size for zero values
    
    # Create the dot-heatmap using transposed and sorted data with size representing rescaled mem_used and color representing ranking with RdYlBu colormap
    ax = fig.add_subplot(fig_sgs[0])
    
    # Plot each point as a dot with marker value set to 'o', edgecolor 'black', and linewidth 0.2
    for x in range(mean_mem_used_ordered_transposed_sorted.shape[1]):
        for y in range(mean_mem_used_ordered_transposed_sorted.shape[0]):
            mean_score = mean_mem_used_ordered_transposed_sorted.iloc[y, x]
            size = dot_size_mem_used_rescaled_small_sorted.iloc[y, x] * 15 + 5 if mean_score != 0 else 5  # Further rescaled and adjusted for log scale
            rank = ranking_mem_used_ordered_transposed_sorted.iloc[y, x]
            color = plt.cm.RdYlBu(rank / ranking_mem_used_ordered_transposed_sorted.max().max())
            ax.scatter(x, y, s=size, color=color, marker='o', edgecolor='black', linewidth=0.2)
    
    # Customize the plot without the internal grid lines
    ax.set_xticks(np.arange(mean_mem_used_ordered_transposed_sorted.shape[1]))
    ax.set_xticklabels(mean_mem_used_ordered_transposed_sorted.columns, rotation=45, horizontalalignment='right', fontsize=8)
    ax.set_yticks(np.arange(mean_mem_used_ordered_transposed_sorted.shape[0]))
    ax.set_yticklabels(mean_mem_used_ordered_transposed_sorted.index, fontsize=8, rotation=45)
    ax.set_xlim(-0.5, mean_mem_used_ordered_transposed_sorted.shape[1] - 0.5)
    ax.set_ylim(-0.5, mean_mem_used_ordered_transposed_sorted.shape[0] - 0.5)
    
    # Remove the grid lines
    ax.grid(False)
    
    # Add the customized legend for mem_used size
    size_patches_rescaled = [plt.scatter([], [], s=size, color='gray', label=f'{value}', marker='o', edgecolor='black', linewidth=0.2) for size, value in zip(legend_sizes, legend_values)]
    legend1 = ax.legend(handles=size_patches_rescaled,
                        title='Memory\nConsumption\n[MB]',
                        loc='upper left', bbox_to_anchor=(1.01, 0.3), frameon=False, scatterpoints=1,
                        fontsize = 8)
    legend1.get_title().set_fontsize(8)

    # Create legend patches for ranking
    unique_ranks = range(1, int(ranking_mem_used_ordered.max().max()) + 1)
    colors = [plt.cm.RdYlBu(rank / ranking_mem_used_ordered.max().max()) for rank in unique_ranks]
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

    fig.add_artist(rank_legend)
    fig.add_artist(legend1)
    axis = fig.add_subplot(ax)

def _generate_subfigure_c(fig: Figure,
                          ax: Axes,
                          gs: GridSpec,
                          subfigure_label: str):

    ax.axis("off")
    _figure_label(ax, subfigure_label, x = -0.18)
    fig_sgs = gs.subgridspec(10,4, wspace = 0.6)

    data = pd.read_csv(os.path.join(DATA_DIR, "Figure2_hyperparameter_tuning_score_data.csv"), index_col = False)
    data = data.groupby(["dataset", "train_size", "algorithm", "tuned", "sample_ID"]).median("f1_score").reset_index()
    data["tuned"] = data["tuned"].map({True: "tuned", False: "not tuned"})
    rev_map = {value: key for key, value in DATASET_MAP.items()}
    data["dataset"] = data["dataset"].map(rev_map)
    data = data.loc[
        data["train_size"] == 5000,
        ["dataset", "train_size", "algorithm", "tuned", "sample_ID", "f1_score"]
    ]

    # Compute the minimum and maximum F1 scores for each dataset and classifier combination
    grouped = data.groupby(['dataset', 'algorithm'])
    min_f1_scores = grouped['f1_score'].min().rename('min_f1_score')
    max_f1_scores = grouped['f1_score'].max().rename('max_f1_score')
    
    # Merge min and max F1 scores back into the original dataframe
    data = data.merge(min_f1_scores, on=['dataset', 'algorithm'])
    data = data.merge(max_f1_scores, on=['dataset', 'algorithm'])
    
    # Normalize the F1 scores within each dataset and classifier combination
    data['normalized_f1_score'] = (data['f1_score'] - data['min_f1_score']) / (data['max_f1_score'] - data['min_f1_score'])
    
    # Compute the median normalized F1 scores for both tuned and untuned classifiers
    medians = data.groupby(['dataset', 'algorithm', 'tuned'])['normalized_f1_score'].median().unstack()
    
    # Compute the actual median F1 scores for coloring
    actual_medians = data.groupby(['dataset', 'algorithm', 'tuned'])['f1_score'].median().unstack()
    
    # Prepare the data for plotting
    datasets = medians.index.get_level_values('dataset').unique()
    classifiers = medians.index.get_level_values('algorithm').unique()
    
    
    # Define a function to plot half-circles with left-right border and outline
    def half_circle(ax, center, radius, angle_start, color, **kwargs):
        theta = np.linspace(angle_start, angle_start + np.pi, 100)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        ax.fill(x, y, color=color, **kwargs)
        ax.plot(x, y, color='black', linewidth=1)  # Add outline
    
    # Define the function to draw separator lines
    def draw_separator_line(ax, center, radius, linewidth=1):
        ax.plot([center[0], center[0]], [center[1] - radius, center[1] + radius], color='black', linewidth=linewidth)
    
    # Prepare the data for plotting
    datasets = medians.index.get_level_values('dataset').unique()
    classifiers = medians.index.get_level_values('algorithm').unique()
    scale_factor = 0.4  # Set the scale factor
    
    # Create subplots with adjusted layout to provide more space for the legend
    ax = fig.add_subplot(fig_sgs[:, 0:3])
    
    # Define the color map
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    norm = plt.Normalize(actual_medians.values.min(), 1)
    
    # Plot half-dots for each classifier and dataset
    for y_pos, classifier in enumerate(classifiers):
        for i, dataset in enumerate(datasets):
            if (dataset, classifier) in medians.index:
                untuned_median = medians.loc[(dataset, classifier), 'not tuned']
                tuned_median = medians.loc[(dataset, classifier), 'tuned']
                untuned_color = cmap(norm(actual_medians.loc[(dataset, classifier), 'not tuned']))
                tuned_color = cmap(norm(actual_medians.loc[(dataset, classifier), 'tuned']))
    
                # Normalize scores to fit the plot scale
                radius_untuned = untuned_median * scale_factor
                radius_tuned = tuned_median * scale_factor
    
                # Plot left half-dot for untuned score
                half_circle(ax, (i, -y_pos), radius_untuned, np.pi/2, untuned_color)
    
                # Plot right half-dot for tuned score
                half_circle(ax, (i, -y_pos), radius_tuned, -np.pi/2, tuned_color)
    
                # Draw separator line
                max_radius = max(radius_untuned, radius_tuned)
                draw_separator_line(ax, (i, -y_pos), max_radius)
    
    # Set labels
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize = 7)
    ax.set_yticks(-np.arange(len(classifiers)))
    ax.set_yticklabels(classifiers, rotation = 45, fontsize = 7)
    
    # Calculate the limits for the rectangle
    rect_x = -0.65
    rect_y = 0.6
    rect_width = len(datasets) + 0.25
    rect_height = -len(classifiers) - 0.25
    
    # Draw a single border around the combined plots area
    rect = Rectangle((rect_x, rect_y), rect_width, rect_height,
                     fill=False, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    
    # Remove individual plot borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add a color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0)
    cbar.set_label('Median F1 Score')
    
    # Create a custom legend
    legend_ax = fig.add_subplot(fig_sgs[4:6,3])#fig.add_axes([0.85, 0.3, 0.2, 0.23])  # Position legend to the right of the color bar with more space
    
    # Plot example half-circles for the legend
    radius = 0.1  # Adjust the radius to fit the legend better
    half_circle(legend_ax, (0.5, 0.6), 0.07, np.pi/2, 'grey')
    half_circle(legend_ax, (0.5, 0.6), radius, -np.pi/2, 'grey')
    draw_separator_line(legend_ax, (0.5, 0.6), radius)
    
    # Add legend labels closer to the circle
    legend_ax.text(0.35, 0.4, 'untuned', va='center', ha='center')
    legend_ax.text(0.65, 0.4, 'tuned', va='center', ha='center')
    
    # Lines from the middle of the semicircles to the labels with increased space between lines and labels
    legend_ax.plot([0.45, 0.35], [0.6, 0.5], color='black', linewidth=1)  # Line to untuned label
    legend_ax.plot([0.55, 0.65], [0.6, 0.5], color='black', linewidth=1)  # Line to tuned label
    
    # Hide axes for legend
    legend_ax.axis('off')

    axis = fig.add_subplot(ax)


def extended_figure_2():

    fig = plt.figure(layout = "constrained",
                     figsize = (FIGURE_WIDTH_FULL, FIGURE_HEIGHT_FULL))

    gs = GridSpec(ncols = 4,
                  nrows = 2,
                  height_ratios = [1.5,1],
                  wspace = 0.7,
                  hspace = 0.4
                  )

    a_coords = gs[0,0]
    b_coords = gs[0,2]
    c_coords = gs[1,:]

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

    plt.savefig("Extended_Figure_2.pdf", dpi = 300, bbox_inches = "tight")
    plt.show()
