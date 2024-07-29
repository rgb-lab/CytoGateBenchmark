from anndata import AnnData
from typing import Optional, Literal, Union
import gc
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
SCALERS = {
    "MinMaxScaler": MinMaxScaler,
    "StandardScaler": StandardScaler
}
def preprocess_data(adata: AnnData,
                    scaling: Optional[Literal["MinMaxScaler", "StandardScaler"]] = "MinMaxScaler",
                    copy: bool = False) -> Optional[AnnData]:
    adata = adata.copy() if copy else adata
    df = adata.to_df(layer = "transformed")
    df = df.clip(lower = np.quantile(df.values, 0.01, axis = 0),
                 upper = np.quantile(df.values, 0.99, axis = 0))
    scaled_array = SCALERS[scaling]().fit_transform(df.values)
    adata.layers["preprocessed"] = scaled_array
    gc.collect()
    return adata if copy else None

def extract_data(adata: AnnData,
                 as_array: bool = False) -> Union[tuple[np.ndarray, np.ndarray],
                                                  tuple[pd.DataFrame, pd.DataFrame]]:
    data_X = adata.to_df(layer = "preprocessed")
    data_y = pd.DataFrame(data = adata.obsm["gating"].toarray(),
                          columns = adata.uns["gating_cols"])
    return (data_X.values, data_y.values) if as_array else (data_X, data_y)