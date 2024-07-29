import pandas as pd
from anndata import AnnData
import FACSPy as fp
import os

def _save_gating(dataset: AnnData,
                 key: str,
                 output_dir) -> None:
    gating = pd.DataFrame(data = dataset.obsm["gating"].toarray(),
                          index = dataset.obs_names,
                          columns = dataset.uns["gating_cols"])
    gating[dataset.obs.columns] = dataset.obs
    if not key.endswith(".csv"):
        key = f"{key}.csv"
    gating.to_csv(os.path.join(output_dir, key))
    return

def _save_dataset(dataset: AnnData,
                  key: str,
                  output_dir: str) -> None:
    fp.save_dataset(dataset,
                    output_dir = output_dir,
                    file_name = key,
                    overwrite = True)
    return

