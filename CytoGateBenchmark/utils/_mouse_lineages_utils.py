import os
from anndata import AnnData
import FACSPy as fp

def _create_dataset(input_directory: str,
                    organ: str) -> AnnData:
    workspace = fp.dt.FlowJoWorkspace(file = os.path.join(input_directory, f"lineages_full_gated_{organ}.wsp"))
    metadata = fp.dt.Metadata(file = os.path.join(input_directory, f"metadata_{organ}.csv"))
    panel = fp.dt.Panel(file = os.path.join(input_directory, "panel.csv"))
    cofactors = fp.dt.CofactorTable(file = os.path.join(input_directory, f"cofactors_{organ}.csv"))

    dataset = fp.create_dataset(input_directory = input_directory,
                                metadata = metadata,
                                panel = panel,
                                workspace = workspace,
                                #subsample_fcs_to = 10_000
                                )
    dataset = dataset[dataset.obs["staining"] == "stained",:].copy()
    dataset = fp.transform(dataset,
                           transform = "asinh",
                           cofactor_table = cofactors,
                           key_added = "transformed",
                           copy = True)
    return dataset
