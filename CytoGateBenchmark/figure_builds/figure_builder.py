from matplotlib import pyplot as plt

from ._config import FIGURE_NAMES

def generate_supervised_figures():
    from ._supervised_characterization import generate_supervised_characterization
    from ._supervised_classifier_characterization import generate_classifier_characterization_fine, generate_classifier_characterization_heatmap

    from ._supervised_characterization_overview import generate_classifier_overview

    # figure_name = FIGURE_NAMES[f"sup_classifier_metrics"]
    # generate_classifier_overview(show = False,
    #                              save = f"figures/{figure_name}.pdf")

    from ._config import MOUSE_LINEAGES_BM_SUPERVISED_FIGURE_KWARGS as FIGURE_KWARGS
    dataset_name = "mouse_lineages_bm"
    generate_supervised_characterization(show = False,
                                         **FIGURE_KWARGS)
    figure_name = FIGURE_NAMES[f"ccomp_{dataset_name}"]
    generate_classifier_characterization_fine(dataset_name = dataset_name,
                                              gates_to_use = FIGURE_KWARGS["gates_to_use"],
                                              show = False,
                                              save = f"figures/{figure_name}.pdf")
    figure_name = FIGURE_NAMES[f"ccomp_hm_{dataset_name}"]
    generate_classifier_characterization_heatmap(dataset_name = dataset_name,
                                                 gates_to_use = None,
                                                 show = False,
                                                 save = f"figures/{figure_name}.pdf")

    from ._config import MOUSE_LINEAGES_PB_SUPERVISED_FIGURE_KWARGS as FIGURE_KWARGS
    dataset_name = "mouse_lineages_pb"
    generate_supervised_characterization(show = False,
                                         **FIGURE_KWARGS)
    figure_name = FIGURE_NAMES[f"ccomp_{dataset_name}"]
    generate_classifier_characterization_fine(dataset_name = dataset_name,
                                              gates_to_use = FIGURE_KWARGS["gates_to_use"],
                                              show = False,
                                              save = f"figures/{figure_name}.pdf")
    figure_name = FIGURE_NAMES[f"ccomp_hm_{dataset_name}"]
    generate_classifier_characterization_heatmap(dataset_name = dataset_name,
                                                 gates_to_use = None,
                                                 show = False,
                                                 save = f"figures/{figure_name}.pdf")

    from ._config import MOUSE_LINEAGES_SPL_SUPERVISED_FIGURE_KWARGS as FIGURE_KWARGS
    dataset_name = "mouse_lineages_spl"
    generate_supervised_characterization(show = False,
                                         **FIGURE_KWARGS)
    figure_name = FIGURE_NAMES[f"ccomp_{dataset_name}"]
    generate_classifier_characterization_fine(dataset_name = dataset_name,
                                              gates_to_use = FIGURE_KWARGS["gates_to_use"],
                                              show = False,
                                              save = f"figures/{figure_name}.pdf")
    figure_name = FIGURE_NAMES[f"ccomp_hm_{dataset_name}"]
    generate_classifier_characterization_heatmap(dataset_name = dataset_name,
                                                 gates_to_use = None,
                                                 show = False,
                                                 save = f"figures/{figure_name}.pdf")


    from ._config import HUMAN_T_CELLS_SUPERVISED_FIGURE_KWARGS as FIGURE_KWARGS
    dataset_name = "human_t_cells"
    generate_supervised_characterization(show = False,
                                         **FIGURE_KWARGS)
    figure_name = FIGURE_NAMES[f"ccomp_{dataset_name}"]
    generate_classifier_characterization_fine(dataset_name = dataset_name,
                                              gates_to_use = FIGURE_KWARGS["gates_to_use"],
                                              show = False,
                                              save = f"figures/{figure_name}.pdf")
    figure_name = FIGURE_NAMES[f"ccomp_hm_{dataset_name}"]
    generate_classifier_characterization_heatmap(dataset_name = dataset_name,
                                                 gates_to_use = None,
                                                 show = False,
                                                 save = f"figures/{figure_name}.pdf")

    from ._config import HIMC_SUPERVISED_FIGURE_KWARGS as FIGURE_KWARGS
    dataset_name = "HIMC"
    generate_supervised_characterization(show = False,
                                         **FIGURE_KWARGS)
    figure_name = FIGURE_NAMES[f"ccomp_{dataset_name}"]
    generate_classifier_characterization_fine(dataset_name = dataset_name,
                                              gates_to_use = FIGURE_KWARGS["gates_to_use"],
                                              show = False,
                                              save = f"figures/{figure_name}.pdf")
    figure_name = FIGURE_NAMES[f"ccomp_hm_{dataset_name}"]
    generate_classifier_characterization_heatmap(dataset_name = dataset_name,
                                                 gates_to_use = None,
                                                 show = False,
                                                 save = f"figures/{figure_name}.pdf")


    from ._config import ZPM_SUPERVISED_FIGURE_KWARGS as FIGURE_KWARGS
    dataset_name = "ZPM"
    generate_supervised_characterization(show = False,
                                         **FIGURE_KWARGS)
    figure_name = FIGURE_NAMES[f"ccomp_{dataset_name}"]
    generate_classifier_characterization_fine(dataset_name = dataset_name,
                                              gates_to_use = FIGURE_KWARGS["gates_to_use"],
                                              show = False,
                                              save = f"figures/{figure_name}.pdf")
    figure_name = FIGURE_NAMES[f"ccomp_hm_{dataset_name}"]
    generate_classifier_characterization_heatmap(dataset_name = dataset_name,
                                                 gates_to_use = None,
                                                 show = False,
                                                 save = f"figures/{figure_name}.pdf")

    
    from ._config import OMIP_SUPERVISED_FIGURE_KWARGS as FIGURE_KWARGS
    dataset_name = "OMIP"
    figure_name = FIGURE_NAMES[f"ccomp_{dataset_name}"]
    generate_supervised_characterization(show = False,
                                         **FIGURE_KWARGS)
    figure_name = FIGURE_NAMES[f"ccomp_{dataset_name}"]
    generate_classifier_characterization_fine(dataset_name = dataset_name,
                                              gates_to_use = FIGURE_KWARGS["gates_to_use"],
                                              show = False,
                                              save = f"figures/{figure_name}.pdf")
    figure_name = FIGURE_NAMES[f"ccomp_hm_{dataset_name}"]
    generate_classifier_characterization_heatmap(dataset_name = dataset_name,
                                                 gates_to_use = None,
                                                 show = False,
                                                 save = f"figures/{figure_name}.pdf")

    
    plt.close("all")
    
    return


def generate_unsupervised_figures():
    from ._unsupervised_characterization import generate_unsupervised_characterization
    from ._unsupervised_classifier_characterization import generate_classifier_characterization

    from ._config import MOUSE_LINEAGES_BM_UNSUPERVISED_FIGURE_KWARGS as FIGURE_KWARGS
    dataset_name = "mouse_lineages_bm"
    figure_name = FIGURE_NAMES[f"algcomp_{dataset_name}"]
    generate_unsupervised_characterization(show = False,
                                           **FIGURE_KWARGS)
    generate_classifier_characterization(dataset_name = dataset_name,
                                         gates_to_use = FIGURE_KWARGS["gates_to_use"],
                                         show = False,
                                         save = f"figures/{figure_name}.pdf")

    from ._config import MOUSE_LINEAGES_PB_UNSUPERVISED_FIGURE_KWARGS as FIGURE_KWARGS
    dataset_name = "mouse_lineages_pb"
    figure_name = FIGURE_NAMES[f"algcomp_{dataset_name}"]
    generate_unsupervised_characterization(show = False,
                                           **FIGURE_KWARGS)
    generate_classifier_characterization(dataset_name = dataset_name,
                                         gates_to_use = FIGURE_KWARGS["gates_to_use"],
                                         show = False,
                                         save = f"figures/{figure_name}.pdf")

    from ._config import MOUSE_LINEAGES_SPL_UNSUPERVISED_FIGURE_KWARGS as FIGURE_KWARGS
    dataset_name = "mouse_lineages_spl"
    figure_name = FIGURE_NAMES[f"algcomp_{dataset_name}"]
    generate_unsupervised_characterization(show = False,
                                           **FIGURE_KWARGS)
    generate_classifier_characterization(dataset_name = dataset_name,
                                         gates_to_use = FIGURE_KWARGS["gates_to_use"],
                                         show = False,
                                         save = f"figures/{figure_name}.pdf")

    from ._config import HUMAN_T_CELLS_UNSUPERVISED_FIGURE_KWARGS as FIGURE_KWARGS
    dataset_name = "human_t_cells"
    figure_name = FIGURE_NAMES[f"algcomp_{dataset_name}"]
    generate_unsupervised_characterization(show = False,
                                           **FIGURE_KWARGS)
    generate_classifier_characterization(dataset_name = dataset_name,
                                         gates_to_use = FIGURE_KWARGS["gates_to_use"],
                                         show = False,
                                         save = f"figures/{figure_name}.pdf")

    from ._config import HIMC_UNSUPERVISED_FIGURE_KWARGS as FIGURE_KWARGS
    dataset_name = "HIMC"
    figure_name = FIGURE_NAMES[f"algcomp_{dataset_name}"]
    generate_unsupervised_characterization(show = False,
                                           **FIGURE_KWARGS)
    generate_classifier_characterization(dataset_name = dataset_name,
                                         gates_to_use = FIGURE_KWARGS["gates_to_use"],
                                         show = False,
                                         save = f"figures/{figure_name}.pdf")

    from ._config import OMIP_UNSUPERVISED_FIGURE_KWARGS as FIGURE_KWARGS
    dataset_name = "OMIP"
    figure_name = FIGURE_NAMES[f"algcomp_{dataset_name}"]
    generate_unsupervised_characterization(show = False,
                                           **FIGURE_KWARGS)
    generate_classifier_characterization(dataset_name = dataset_name,
                                         gates_to_use = FIGURE_KWARGS["gates_to_use"],
                                         show = False,
                                         save = f"figures/{figure_name}.pdf")

    from ._config import ZPM_UNSUPERVISED_FIGURE_KWARGS as FIGURE_KWARGS
    dataset_name = "ZPM"
    figure_name = FIGURE_NAMES[f"algcomp_{dataset_name}"]
    generate_unsupervised_characterization(show = False,
                                           **FIGURE_KWARGS)
    generate_classifier_characterization(dataset_name = dataset_name,
                                         gates_to_use = FIGURE_KWARGS["gates_to_use"],
                                         show = False,
                                         save = f"figures/{figure_name}.pdf")
 
    plt.close("all")
    return

def generate_jaccard_figures():
    from ._jaccard import generate_jaccard_comparison

    from ._config import HUMAN_T_CELLS_JACCARD_FIGURE_KWARGS as FIGURE_KWARGS
    dataset_name = "human_t_cells"
    for cell_type in ["CD4_CM", "CD4_EM", "CD4_TEMRA", "CD4_Naive"]:
        figure_name = FIGURE_NAMES[f"jaccard_{dataset_name}_{cell_type}"]
        generate_jaccard_comparison(**FIGURE_KWARGS,
                                    cell_type = cell_type,
                                    base_population = "CD4+_T-cells",
                                    population_to_show = "CD4+_T-cells",
                                    save = f"figures/{figure_name}.pdf")

    for cell_type in ["CD8_CM", "CD8_EM", "CD8_TEMRA", "CD8_Naive"]:
        figure_name = FIGURE_NAMES[f"jaccard_{dataset_name}_{cell_type}"]
        generate_jaccard_comparison(**FIGURE_KWARGS,
                                    cell_type = cell_type,
                                    base_population = "CD8+_T-cells",
                                    population_to_show = "CD8+_T-cells",
                                    save = f"figures/{figure_name}.pdf")

    
    from ._config import MOUSE_LINEAGES_BM_JACCARD_FIGURE_KWARGS as FIGURE_KWARGS
    dataset_name = "mouse_lineages_bm"
    for cell_type in ["T_cells", "NK_cells"]:
        figure_name = FIGURE_NAMES[f"jaccard_{dataset_name}_{cell_type}"]
        generate_jaccard_comparison(**FIGURE_KWARGS,
                                    cell_type = cell_type,
                                    save = f"figures/{figure_name}.pdf")

    plt.close("all")
    return

def generate_figures():
    generate_supervised_figures()
    generate_unsupervised_figures()
    generate_jaccard_figures()

    
