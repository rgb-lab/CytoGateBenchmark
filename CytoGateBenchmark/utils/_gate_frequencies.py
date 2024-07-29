import os
from anndata import AnnData
import pandas as pd

def generate_gate_frequencies():
    from ._human_t_cells_utils import _create_dataset
    dataset = _create_dataset(input_directory = "datasets/human_T_cells/")
    _create_gating_frame(dataset, "human_t_cells")

    from ._mouse_lineages_utils import _create_dataset
    dataset = _create_dataset(input_directory = "datasets/mouse_lineages/",
                              organ = "bm")
    _create_gating_frame(dataset, "mouse_lineages_bm")
    dataset = _create_dataset(input_directory = "datasets/mouse_lineages/",
                              organ = "pb")
    _create_gating_frame(dataset, "mouse_lineages_pb")
    dataset = _create_dataset(input_directory = "datasets/mouse_lineages/",
                              organ = "spl")
    _create_gating_frame(dataset, "mouse_lineages_spl")

    from ._OMIP_utils import _create_dataset
    dataset = _create_dataset(input_directory = "datasets/OMIP")
    _create_gating_frame(dataset, "OMIP")
    
    from ._ZPM_utils import _create_dataset
    dataset = _create_dataset(input_directory = "datasets/ZPM")
    _create_gating_frame(dataset, "ZPM")

    from ._HIMC_utils import _create_dataset
    dataset = _create_dataset(input_directory = "datasets/HIMC")
    _create_gating_frame(dataset, "HIMC")

    return

def _create_gating_frame(dataset: AnnData,
                         dataset_name: str) -> None:
    gating = pd.DataFrame(data = dataset.obsm["gating"].toarray(),
                          columns = dataset.uns["gating_cols"],
                          index = dataset.obs_names)
    gating["sample_ID"] = dataset.obs["sample_ID"]
    counts = pd.DataFrame(gating.groupby("sample_ID").sum())
    freqs = pd.DataFrame(gating.groupby("sample_ID").sum() / gating.groupby("sample_ID").count())

    if not os.path.exists(os.path.join(os.getcwd(), "gate_counts")):
        os.mkdir(os.path.join(os.getcwd(), "gate_counts"))

    counts.to_csv(f"gate_counts/{dataset_name}_counts.csv")
    freqs.to_csv(f"gate_counts/{dataset_name}_freqs.csv")
    print("Gating frequencies saved successfully")

    return
