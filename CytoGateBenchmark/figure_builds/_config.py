from ._utils import SUPERVISED_UMAP_PALETTE

FIGURE_NAMES = {

    "sup_classifier_metrics": "Figure_1",

    "ccomp_hm_mouse_lineages_bm": "Supplementary_Figure_1",
    "ccomp_hm_mouse_lineages_pb": "Supplementary_Figure_2",
    "ccomp_hm_mouse_lineages_spl": "Supplementary_Figure_3",
    "ccomp_hm_human_t_cells": "Supplementary_Figure_4",
    "ccomp_hm_OMIP": "Supplementary_Figure_5",
    "ccomp_hm_ZPM": "Supplementary_Figure_6",
    "ccomp_hm_HIMC": "Supplementary_Figure_7",

    "sup_mouse_lineages_bm": "Supplementary_Figure_8",
    "sup_mouse_lineages_pb": "Supplementary_Figure_9",
    "sup_mouse_lineages_spl": "Supplementary_Figure_10",
    "sup_human_t_cells": "Supplementary_Figure_11",
    "sup_OMIP": "Supplementary_Figure_12",
    "sup_ZPM": "Figure_2",
    "sup_HIMC": "Supplementary_Figure_13",

    "ccomp_mouse_lineages_bm": "Supplementary_Figure_XX1",
    "ccomp_mouse_lineages_pb": "Supplementary_Figure_XX2",
    "ccomp_mouse_lineages_spl": "Supplementary_Figure_XX3",
    "ccomp_human_t_cells": "Supplementary_Figure_XX4",
    "ccomp_OMIP": "Supplementary_Figure_XX5",
    "ccomp_ZPM": "Supplementary_Figure_XX6",
    "ccomp_HIMC": "Supplementary_Figure_XX7",

    "jaccard_mouse_lineages_bm_T_cells": "Extended_Data_Figure_3",
    "jaccard_human_t_cells_CD4_CM": "Supplementary_Figure_14",

    "unsup_mouse_lineages_bm": "Supplementary_Figure_15",
    "unsup_mouse_lineages_pb": "Figure_3",
    "unsup_mouse_lineages_spl": "Supplementary_Figure_16",
    "unsup_human_t_cells": "Supplementary_Figure_17",
    "unsup_OMIP": "Supplementary_Figure_18",
    "unsup_ZPM": "Supplementary_Figure_19",
    "unsup_HIMC": "Supplementary_Figure_20",

    "algcomp_mouse_lineages_bm": "Supplementary_Figure_21",
    "algcomp_mouse_lineages_pb": "Supplementary_Figure_22",
    "algcomp_mouse_lineages_spl": "Supplementary_Figure_23",
    "algcomp_human_t_cells": "Supplementary_Figure_24",
    "algcomp_OMIP": "Supplementary_Figure_25",
    "algcomp_ZPM": "Supplementary_Figure_26",
    "algcomp_HIMC": "Supplementary_Figure_27",

    "jaccard_human_t_cells_CD4_EM": "Supplementary_Figure_XX8",
    "jaccard_human_t_cells_CD4_TEMRA": "Supplementary_Figure_XX9",
    "jaccard_human_t_cells_CD4_Naive": "Supplementary_Figure_XX10",

    "jaccard_human_t_cells_CD8_CM": "Supplementary_Figure_XX11",
    "jaccard_human_t_cells_CD8_EM": "Supplementary_Figure_XX12",
    "jaccard_human_t_cells_CD8_TEMRA": "Supplementary_Figure_XX13",
    "jaccard_human_t_cells_CD8_Naive": "Supplementary_Figure_XX14",
    "jaccard_mouse_lineages_bm_NK_cells": "Supplementary_Figure_XX15",
}

MOUSE_LINEAGES_BM_SUPERVISED_FIGURE_KWARGS = {
    "dataset_name": "mouse_lineages_bm",
    "population_to_show": "CD45+",
    "train_sample_IDs": ["3", "4"],
    "gates_to_use": [
        "root/cells/singlets/live/CD45+/B_cells",
        "root/cells/singlets/live/CD45+/T_cells",
        "root/cells/singlets/live/CD45+/NK_cells",
        "root/cells/singlets/live/CD45+/CD11b+/Eosinophils",
        "root/cells/singlets/live/CD45+/CD11b+/Monocytes",
        "root/cells/singlets/live/CD45+/CD11b+/Neutrophils",
        "root/cells/singlets/live/CD45+/T_cells/CD4_T_cells",
        "root/cells/singlets/live/CD45+/T_cells/CD8_T_cells",
        "root/cells/singlets/live/CD45+/T_cells/DP_T_cells"
    ],
    "palette": SUPERVISED_UMAP_PALETTE,
    "pca_kwargs": {"exclude": "GFP", "scaling": None},
    "neighbors_kwargs": {"exclude": "GFP", "scaling": None},
    "umap_kwargs": {"exclude": "GFP", "scaling": None},
    "biax_layout_kwargs": {"xlim_0": -500},
    "biax_kwargs": {"gate": "CD45+",
                    "x_channel": "B220",
                    "y_channel": "SSC-A",
                    "sample_identifier": "5",
                    "layer": "compensated",
                    "show": False},
    "wsp_group": "bone_marrow",
    "wsp_file": "20112023_lineage_BM_M3_040.fcs",
    "graphical_abstract_gate": "B_cells",
    "classifier": "DecisionTreeClassifier",
    "save": f"figures/{FIGURE_NAMES['sup_mouse_lineages_bm']}.pdf"
    
}

MOUSE_LINEAGES_SPL_SUPERVISED_FIGURE_KWARGS = {
    "dataset_name": "mouse_lineages_spl",
    "population_to_show": "CD45+",
    "train_sample_IDs": ["3", "4"],
    "gates_to_use": [
        "root/cells/singlets/live/CD45+/B_cells",
        "root/cells/singlets/live/CD45+/T_cells",
        "root/cells/singlets/live/CD45+/NK_cells",
        "root/cells/singlets/live/CD45+/CD11b+/Eosinophils",
        "root/cells/singlets/live/CD45+/CD11b+/Monocytes",
        "root/cells/singlets/live/CD45+/CD11b+/Neutrophils",
        "root/cells/singlets/live/CD45+/T_cells/CD4_T_cells",
        "root/cells/singlets/live/CD45+/T_cells/CD8_T_cells",
        "root/cells/singlets/live/CD45+/T_cells/DP_T_cells"
    ],
    "palette": SUPERVISED_UMAP_PALETTE,
    "pca_kwargs": {"exclude": "GFP", "scaling": None},
    "neighbors_kwargs": {"exclude": "GFP", "scaling": None},
    "umap_kwargs": {"exclude": "GFP", "scaling": None},
    "biax_layout_kwargs": {"xlim_0": -500},
    "biax_kwargs": {"gate": "CD45+",
                    "x_channel": "B220",
                    "y_channel": "SSC-A",
                    "sample_identifier": "5",
                    "layer": "compensated",
                    "show": False},
    "wsp_group": "spleen",
    "wsp_file": "20112023_lineage_SPL_M3_048.fcs",
    "graphical_abstract_gate": "B_cells",
    "classifier": "DecisionTreeClassifier",
    "save": f"figures/{FIGURE_NAMES['sup_mouse_lineages_spl']}.pdf"
    
}

MOUSE_LINEAGES_PB_SUPERVISED_FIGURE_KWARGS = {
    "dataset_name": "mouse_lineages_pb",
    "population_to_show": "CD45+",
    "train_sample_IDs": ["3", "4"],
    "gates_to_use": [
        "root/cells/singlets/live/CD45+/B_cells",
        "root/cells/singlets/live/CD45+/T_cells",
        "root/cells/singlets/live/CD45+/NK_cells",
        "root/cells/singlets/live/CD45+/CD11b+/Eosinophils",
        "root/cells/singlets/live/CD45+/CD11b+/Monocytes",
        "root/cells/singlets/live/CD45+/CD11b+/Neutrophils",
        "root/cells/singlets/live/CD45+/T_cells/CD4_T_cells",
        "root/cells/singlets/live/CD45+/T_cells/CD8_T_cells",
        "root/cells/singlets/live/CD45+/T_cells/DP_T_cells"
    ],
    "palette": SUPERVISED_UMAP_PALETTE,
    "pca_kwargs": {"exclude": "GFP", "scaling": None},
    "neighbors_kwargs": {"exclude": "GFP", "scaling": None},
    "umap_kwargs": {"exclude": "GFP", "scaling": None},
    "biax_layout_kwargs": {"xlim_0": -500},
    "biax_kwargs": {"gate": "CD45+",
                    "x_channel": "B220",
                    "y_channel": "SSC-A",
                    "sample_identifier": "5",
                    "layer": "compensated",
                    "show": False},
    "wsp_group": "peripheral_blood",
    "wsp_file": "20112023_lineage_PB_M3_032.fcs",
    "graphical_abstract_gate": "B_cells",
    "classifier": "DecisionTreeClassifier",
    "save": f"figures/{FIGURE_NAMES['sup_mouse_lineages_pb']}.pdf"
    
}

HUMAN_T_CELLS_SUPERVISED_FIGURE_KWARGS = {
    "dataset_name": "human_t_cells",
    "population_to_show": "Lymphocytes",
    "train_sample_IDs": ["4", "5", "6"],
    "gates_to_use": [
        "root/Singlets/Lymphocytes/T_cells",
        "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells",
        "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells",
        "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells/CD4_CM",
        "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells/CD4_EM",
        "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells/CD4_Naive",
        "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells/CD8_CM",
        "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells/CD8_EM",
        "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells/CD8_Naive"
    ],
    "palette": SUPERVISED_UMAP_PALETTE,
    "pca_kwargs": {"scaling": None},
    "neighbors_kwargs": {"scaling": None},
    "umap_kwargs": {"scaling": None},
    "biax_layout_kwargs": {"xlim_0": -2000},
    "biax_kwargs": {"gate": "Lymphocytes",
                    "x_channel": "CD3",
                    "y_channel": "SSC-A",
                    "sample_identifier": "2",
                    "layer": "compensated",
                    "show": False},
    "wsp_group": "All Samples",
    "wsp_file": "Sopro_TA_030722_AS_140622.fcs",
    "graphical_abstract_gate": "T_cells",
    "classifier": "DecisionTreeClassifier",
    "save": f"figures/{FIGURE_NAMES['sup_human_t_cells']}.pdf"

}

HIMC_SUPERVISED_FIGURE_KWARGS = {
    "dataset_name": "HIMC",
    "population_to_show": "live",
    "train_sample_IDs": ["1", "2", "3"],
    "gates_to_use": [
        "root/beads_negative/cells/singlets/live/CD3+",
        "root/beads_negative/cells/singlets/live/CD3+/CD4_T_cells",
        "root/beads_negative/cells/singlets/live/CD3+/CD8+_T_cells",
        "root/beads_negative/cells/singlets/live/CD3+/DN_T_cells",
        "root/beads_negative/cells/singlets/live/CD3+/DP_T_cells",
        "root/beads_negative/cells/singlets/live/CD3-/B_cells",
        "root/beads_negative/cells/singlets/live/CD3-/CD3-_HLADR-/NK_cells",
        "root/beads_negative/cells/singlets/live/CD3-/CD19-CD56-/DC",
        "root/beads_negative/cells/singlets/live/CD3-/TCRgd+"
    ],
    "palette": SUPERVISED_UMAP_PALETTE,
    "pca_kwargs": {"scaling": None},
    "neighbors_kwargs": {"scaling": None},
    "umap_kwargs": {"scaling": None},
    "biax_layout_kwargs": {"xlim_0": -10},
    "biax_kwargs": {"gate": "live",
                    "x_channel": "CD3",
                    "y_channel": "TCRgd",
                    "sample_identifier": "5",
                    "layer": "compensated",
                    "show": False},
    "wsp_group": "All Samples",
    "wsp_file": "081216-Mike-HIMC ctrls-001_01_normalized.fcs",
    "graphical_abstract_gate": "CD3+",
    "classifier": "DecisionTreeClassifier",
    "save": f"figures/{FIGURE_NAMES['sup_HIMC']}.pdf"
    
}

ZPM_SUPERVISED_FIGURE_KWARGS = {
    "dataset_name": "ZPM",
    "population_to_show": "Lymphocytes",
    "train_sample_IDs": ["1", "3", "5", "7", "9", "11", "13", "15"],
    "gates_to_use": [
#        "root/singlets/cells",
#        "root/singlets/cells/live",
#        "root/singlets/cells/live/Lymphocytes",
        "root/singlets/cells/live/Lymphocytes/CD3+",
        "root/singlets/cells/live/Lymphocytes/CD3+/CD4+",
        "root/singlets/cells/live/Lymphocytes/CD3+/CD4+/TH1-like",
        "root/singlets/cells/live/Lymphocytes/CD3+/CD4+/TH1-like/TH1_activated",
        "root/singlets/cells/live/Lymphocytes/CD3+/CD4+/TH17-like",
        "root/singlets/cells/live/Lymphocytes/CD3+/CD4+/TH17-like/TH17_activated"
    ],
    "palette": SUPERVISED_UMAP_PALETTE,
    "pca_kwargs": {"scaling": None},
    "neighbors_kwargs": {"scaling": None},
    "umap_kwargs": {"scaling": None},
    "biax_layout_kwargs": {"xlim_0": -1000},
    "biax_kwargs": {"gate": "live",
                    "x_channel": "CD3",
                    "y_channel": "SSC-A",
                    "sample_identifier": "17",
                    "layer": "compensated",
                    "show": False},
    "wsp_group": "All Samples",
    "wsp_file": "20220104_Specimen_001_1-34-A.fcs",
    "graphical_abstract_gate": "CD3+",
    "classifier": "DecisionTreeClassifier",
    "save": f"figures/{FIGURE_NAMES['sup_ZPM']}.pdf"
}

OMIP_SUPERVISED_FIGURE_KWARGS = {
    "dataset_name": "OMIP",
    "population_to_show": "CD45+",
    "train_sample_IDs": ["1"],
    "gates_to_use": [
        "root/singlets/live/CD45+/CD3+_TCRgd-/CD3+_CD56-",
        "root/singlets/live/CD45+/CD3+_TCRgd-/CD3+_CD56-/CD4+_T_cells",
        "root/singlets/live/CD45+/CD3+_TCRgd-/CD3+_CD56-/CD8+_T_cells",
        "root/singlets/live/CD45+/CD3+_TCRgd-/CD3+_CD56-/DN_T_cells",
        "root/singlets/live/CD45+/CD3+_TCRgd-/CD3+_CD56-/DP_T_cells",
        "root/singlets/live/CD45+/CD3-/B_cells",
        "root/singlets/live/CD45+/CD3-/NK_cells",
        "root/singlets/live/CD45+/CD3-/Monocytes",
        "root/singlets/live/CD45+/CD3+_TCRgd+"
    ],
    "palette": SUPERVISED_UMAP_PALETTE,
    "pca_kwargs": {"scaling": None},
    "neighbors_kwargs": {"scaling": None},
    "umap_kwargs": {"scaling": None},
    "biax_layout_kwargs": {"xlim_0": -5000},
    "biax_kwargs": {"gate": "CD3+_CD56-",
                    "x_channel": "CD4",
                    "y_channel": "CD8",
                    "sample_identifier": "2",
                    "layer": "compensated",
                    "show": False},
    "wsp_group": "All Samples",
    "wsp_file": "MC 303444.fcs",
    "graphical_abstract_gate": "CD4+_T_cells",
    "classifier": "DecisionTreeClassifier",
    "save": f"figures/{FIGURE_NAMES['sup_OMIP']}.pdf"
    
}

MOUSE_LINEAGES_BM_UNSUPERVISED_FIGURE_KWARGS = {
    "dataset_name": "mouse_lineages_bm",
    "algorithm": "parc",
    "population_to_show": "CD45+",
    "gates_to_use": [
        "root/cells/singlets/live/CD45+/B_cells",
        "root/cells/singlets/live/CD45+/T_cells",
        "root/cells/singlets/live/CD45+/NK_cells",
        "root/cells/singlets/live/CD45+/CD11b+/Eosinophils",
        "root/cells/singlets/live/CD45+/CD11b+/Monocytes",
        "root/cells/singlets/live/CD45+/CD11b+/Neutrophils",
        "root/cells/singlets/live/CD45+/T_cells/CD4_T_cells",
        "root/cells/singlets/live/CD45+/T_cells/CD8_T_cells"
    ],
    "palette": SUPERVISED_UMAP_PALETTE,
    "pca_kwargs": {"exclude": "GFP", "scaling": None},
    "neighbors_kwargs": {"exclude": "GFP", "scaling": None},
    "umap_kwargs": {"exclude": "GFP", "scaling": None},
    "clustering_kwargs": {"resolution_parameter": 0.8},
    "graphical_abstract_markers": ["B220", "CD45", "Ly6G", "CD3"],
    "umap_markers": ["CD3", "CD4", "CD8", "B220", "NK1.1", "Siglec_F", "Ly6G", "Ly6C", "CD11b"],
    "vmax_map": {
        "CD3": 2.5,
        "CD4": 2.5,
        "CD8": 2.5,
        "B220": 4,
        "NK1.1": 2.5,
        "Siglec_F": 2.5,
        "Ly6G": 3,
        "Ly6C": 4.5,
        "CD11b": 4.5
    }, 
    "graphical_abstract_gate": "B_cells",
    "save": f"figures/{FIGURE_NAMES['unsup_mouse_lineages_bm']}.pdf"

}

MOUSE_LINEAGES_PB_UNSUPERVISED_FIGURE_KWARGS = {
    "dataset_name": "mouse_lineages_pb",
    "algorithm": "parc",
    "population_to_show": "CD45+",
    "gates_to_use": [
        "root/cells/singlets/live/CD45+/B_cells",
        "root/cells/singlets/live/CD45+/T_cells",
        "root/cells/singlets/live/CD45+/NK_cells",
        "root/cells/singlets/live/CD45+/CD11b+/Eosinophils",
        "root/cells/singlets/live/CD45+/CD11b+/Monocytes",
        "root/cells/singlets/live/CD45+/CD11b+/Neutrophils",
        "root/cells/singlets/live/CD45+/T_cells/CD4_T_cells",
        "root/cells/singlets/live/CD45+/T_cells/CD8_T_cells"
    ],
    "palette": SUPERVISED_UMAP_PALETTE,
    "pca_kwargs": {"exclude": "GFP", "scaling": None},
    "neighbors_kwargs": {"exclude": "GFP", "scaling": None},
    "umap_kwargs": {"exclude": "GFP", "scaling": None},
    "clustering_kwargs": {"resolution_parameter": 0.8},
    "graphical_abstract_markers": ["B220", "CD45", "Ly6G", "CD3"],
    "umap_markers": ["CD3", "CD4", "CD8", "B220", "NK1.1", "Siglec_F", "Ly6G", "Ly6C", "CD11b"],
    "vmax_map": {
        "CD3": 2.5,
        "CD4": 2.5,
        "CD8": 2.5,
        "B220": 4,
        "NK1.1": 2.5,
        "Siglec_F": 2.5,
        "Ly6G": 3,
        "Ly6C": 4.5,
        "CD11b": 4.5
    }, 
    "graphical_abstract_gate": "B_cells",
    "save": f"figures/{FIGURE_NAMES['unsup_mouse_lineages_pb']}.pdf"
}

MOUSE_LINEAGES_SPL_UNSUPERVISED_FIGURE_KWARGS = {
    "dataset_name": "mouse_lineages_spl",
    "algorithm": "parc",
    "population_to_show": "CD45+",
    "gates_to_use": [
        "root/cells/singlets/live/CD45+/B_cells",
        "root/cells/singlets/live/CD45+/T_cells",
        "root/cells/singlets/live/CD45+/NK_cells",
        "root/cells/singlets/live/CD45+/CD11b+/Eosinophils",
        "root/cells/singlets/live/CD45+/CD11b+/Monocytes",
        "root/cells/singlets/live/CD45+/CD11b+/Neutrophils",
        "root/cells/singlets/live/CD45+/T_cells/CD4_T_cells",
        "root/cells/singlets/live/CD45+/T_cells/CD8_T_cells"
    ],
    "palette": SUPERVISED_UMAP_PALETTE,
    "pca_kwargs": {"exclude": "GFP", "scaling": None},
    "neighbors_kwargs": {"exclude": "GFP", "scaling": None},
    "umap_kwargs": {"exclude": "GFP", "scaling": None},
    "clustering_kwargs": {"resolution_parameter": 0.8},
    "graphical_abstract_markers": ["B220", "CD45", "Ly6G", "CD3"],
    "umap_markers": ["CD3", "CD4", "CD8", "B220", "NK1.1", "Siglec_F", "Ly6G", "Ly6C", "CD11b"],
    "vmax_map": {
        "CD3": 2.5,
        "CD4": 2.5,
        "CD8": 2.5,
        "B220": 4,
        "NK1.1": 2.5,
        "Siglec_F": 2.5,
        "Ly6G": 3,
        "Ly6C": 4.5,
        "CD11b": 4.5
    }, 
    "graphical_abstract_gate": "B_cells",
    "save": f"figures/{FIGURE_NAMES['unsup_mouse_lineages_spl']}.pdf"
}

HUMAN_T_CELLS_UNSUPERVISED_FIGURE_KWARGS = {
    "dataset_name": "human_t_cells",
    "algorithm": "parc",
    "population_to_show": "Lymphocytes",
    "gates_to_use": [
        "root/Singlets/Lymphocytes/T_cells",
        "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells",
        "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells",
        "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells/CD4_Naive",
        "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells/CD4_CM",
        "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells/CD4_EM",
        "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells/CD8_Naive",
        "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells/CD8_CM",
        "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells/CD8_EM",
    ],
    "palette": SUPERVISED_UMAP_PALETTE,
    "pca_kwargs": {"scaling": None},
    "neighbors_kwargs": {"scaling": None},
    "umap_kwargs": {"scaling": None},
    "clustering_kwargs": {"resolution_parameter": 0.8},
    "graphical_abstract_markers": ["CD3", "CD4", "CD8"],
    "umap_markers": ["CD3", "CD4", "CD8", "CD197", "CD45RA", "CD28", "CD57", "HLA_DR", "TCR_g_d_APC_R700"],
    "vmax_map": {
        "CD3": 5,
        "CD4": 5,
        "CD8": 5,
        "CD197": 3,
        "CD45RA": 4,
        "CD28": 4,
        "CD57": 3,
        "HLA_DR": 4.5,
        "TCR_g_d_APC_R700": 1.5
    }, 
    "graphical_abstract_gate": "CD4+_T-cells",
    "save": f"figures/{FIGURE_NAMES['unsup_human_t_cells']}.pdf"
}

HIMC_UNSUPERVISED_FIGURE_KWARGS = {
    "dataset_name": "HIMC",
    "algorithm": "parc",
    "population_to_show": "live",
    "gates_to_use": [
        "root/beads_negative/cells/singlets/live/CD3+",
        "root/beads_negative/cells/singlets/live/CD3+/CD4_T_cells",
        "root/beads_negative/cells/singlets/live/CD3+/CD8+_T_cells",
        "root/beads_negative/cells/singlets/live/CD3-/TCRgd+",
        "root/beads_negative/cells/singlets/live/CD3-/B_cells",
        "root/beads_negative/cells/singlets/live/CD3-/CD3-_HLADR-/NK_cells",
        "root/beads_negative/cells/singlets/live/CD3-/CD19-CD56-/DC",
        "root/beads_negative/cells/singlets/live/CD3+/DN_T_cells",
        "root/beads_negative/cells/singlets/live/CD3+/DP_T_cells",
    ],
    "palette": SUPERVISED_UMAP_PALETTE,
    "pca_kwargs": {"scaling": None},
    "neighbors_kwargs": {"scaling": None},
    "umap_kwargs": {"scaling": None},
    "clustering_kwargs": {"resolution_parameter": 0.8},
    "graphical_abstract_markers": ["CD3", "CD4", "CD8", "CD19"],
    "umap_markers": ["CD3", "CD4", "CD8", "CCR7", "CD45RA", "CD28", "CD57", "HLADR", "TCRgd"],
    "vmax_map": {
        "CD3": 3,
        "CD4": 3.5,
        "CD8": 3.5,
        "CCR7": 3,
        "CD45RA": 4,
        "CD28": 4,
        "CD57": 3.5,
        "HLADR": 4.5,
        "TCRgd": 1.5
    }, 
    "graphical_abstract_gate": "CD8+_T_cells",
    "save": f"figures/{FIGURE_NAMES['unsup_HIMC']}.pdf"
}

OMIP_UNSUPERVISED_FIGURE_KWARGS = {
    "dataset_name": "OMIP",
    "algorithm": "parc",
    "population_to_show": "CD45+",
    "gates_to_use": [
        "root/singlets/live/CD45+/CD3+_TCRgd-/CD3+_CD56-",
        "root/singlets/live/CD45+/CD3+_TCRgd-/CD3+_CD56-/CD4+_T_cells",
        "root/singlets/live/CD45+/CD3+_TCRgd-/CD3+_CD56-/CD8+_T_cells",
        "root/singlets/live/CD45+/CD3-/B_cells",
        "root/singlets/live/CD45+/CD3-/NK_cells",
        "root/singlets/live/CD45+/CD3+_TCRgd+",
        "root/singlets/live/CD45+/CD3-/CD19neg_CD56neg/Lin-_HLADR+/CD11c+_DC",
        "root/singlets/live/CD45+/CD3+_TCRgd-/CD3+_CD56-/DN_T_cells",
        "root/singlets/live/CD45+/CD3+_TCRgd-/CD3+_CD56-/DP_T_cells"
    ],
    "palette": SUPERVISED_UMAP_PALETTE,
    "pca_kwargs": {"scaling": None},
    "neighbors_kwargs": {"scaling": None},
    "umap_kwargs": {"scaling": None},
    "clustering_kwargs": {"resolution_parameter": 0.8},
    "graphical_abstract_markers": ["CD3", "CD4", "CD8", "CD20"],
    "umap_markers": ["CD3", "CD4", "CD8", "CCR7", "CD45RA", "CD28", "CD57", "HLADR", "TCRgd"],
    "vmax_map": {
        "CD3": 5,
        "CD4": 5,
        "CD8": 5,
        "CCR7": 3,
        "CD45RA": 4,
        "CD28": 4,
        "CD57": 3,
        "HLADR": 4.5,
        "TCRgd": 1.5
    }, 
    "graphical_abstract_gate": "CD4+_T_cells",
    "save": f"figures/{FIGURE_NAMES['unsup_OMIP']}.pdf"
}

ZPM_UNSUPERVISED_FIGURE_KWARGS = {
    "dataset_name": "ZPM",
    "algorithm": "parc",
    "population_to_show": "Lymphocytes",
    "gates_to_use": [
#        "root/singlets/cells",
#        "root/singlets/cells/live",
#        "root/singlets/cells/live/Lymphocytes",
        "root/singlets/cells/live/Lymphocytes/CD3+",
        "root/singlets/cells/live/Lymphocytes/CD3+/CD4+",
        "root/singlets/cells/live/Lymphocytes/CD3+/CD4+/TH1-like",
        "root/singlets/cells/live/Lymphocytes/CD3+/CD4+/TH1-like/TH1_activated",
        "root/singlets/cells/live/Lymphocytes/CD3+/CD4+/TH17-like",
        "root/singlets/cells/live/Lymphocytes/CD3+/CD4+/TH17-like/TH17_activated"
    ],
    "palette": SUPERVISED_UMAP_PALETTE,
    "pca_kwargs": {"scaling": None},
    "neighbors_kwargs": {"scaling": None},
    "umap_kwargs": {"scaling": None},
    "clustering_kwargs": {"resolution_parameter": 0.8},
    "graphical_abstract_markers": ["CD3", "CD4"],
    "umap_markers": ["CD3", "CD4", "CXCR3", "CCR6", "HLADR", "CD38"],
    "vmax_map": {
        "CD3": 5,
        "CD4": 5,
        "CXCR3": 5,
        "CCR6": 3,
        "HLADR": 4,
        "CD38": 4,
    }, 
    "graphical_abstract_gate": "CD3+",
    "save": f"figures/{FIGURE_NAMES['unsup_ZPM']}.pdf"
}


MOUSE_LINEAGES_BM_JACCARD_FIGURE_KWARGS = {
    "dataset_name": "mouse_lineages_bm",
    "flowjo_plot_directory": "figure_data/jaccard/mouse_lineages_bm",
    "base_population": "CD45+",
    "dimred_params": {"scaling": None, "exclude": ["GFP"]},
    "umap_markers": ["CD3", "CD4", "CD8", "B220", "NK1.1", "Siglec_F", "Ly6G", "Ly6C", "CD11b"],
    "vmax_map": {
        "CD3": 3,
        "CD4": 2.5,
        "CD8": 2.5,
        "B220": 5,
        "NK1.1": 2.5,
        "Siglec_F": 2.5,
        "Ly6G": 3,
        "Ly6C": 4.5,
        "CD11b": 4.5
    },
    "show": False
}


HUMAN_T_CELLS_JACCARD_FIGURE_KWARGS = {
    "dataset_name": "human_t_cells",
    "flowjo_plot_directory": "figure_data/jaccard/human_T_cells",
    "dimred_params": {"scaling": None},
    "umap_markers": ["CD3", "CD4", "CD8", "CD197", "CD45RA", "CD28", "CD57", "HLA_DR", "TCR_g_d_APC_R700"],
    "vmax_map": {
        "CD3": 5,
        "CD4": 5,
        "CD8": 5,
        "CD197": 5,
        "CD45RA": 4.5,
        "CD28": 6,
        "CD57": 3.5,
        "HLA_DR": 4.5,
        "TCR_g_d_APC_R700": 1.5
    },
    "show": False
}
