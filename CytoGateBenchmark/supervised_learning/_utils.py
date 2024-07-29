import numpy as np
from anndata import AnnData
import gc

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from FACSPy.model._sampling import GateSampler

SAMPLER_KWARGS = {
    "oversampler": "Gaussian",
    "target_size_per_gate": None,
    "oversample_rare_cells": True,
    "rare_cells_cutoff": 1000,
    "rare_cells_target_size_per_gate": None,
    "rare_cells_target_fraction": 0.01
}

SCALER = StandardScaler()

def _subsample_arrays(X: np.ndarray,
                      y: np.ndarray,
                      n: int) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(187)
    if n > X.shape[0]:
        return X, y
    idxs = np.random.choice(range(X.shape[0]), n, replace=False)
    return X[idxs], y[idxs]


def _use_sampler(adata: AnnData,
                 train_size: int,
                 test_size: float) -> tuple[np.ndarray,
                                            np.ndarray,
                                            np.ndarray,
                                            np.ndarray,
                                            MinMaxScaler]:
    scaler = MinMaxScaler()
    scaler.fit(adata.layers["transformed"])
    adata.layers["preprocessed"] = scaler.transform(adata.layers["transformed"])

    ### NEW SAMPLER LOGIC
    sampler = GateSampler(oversampler = "Gaussian",
                          target_size = train_size,
                          target_size_per_gate = None,
                          oversample_rare_cells = True,
                          rare_cells_cutoff = 100,
                          rare_cells_target_size_per_gate = None,
                          rare_cells_target_fraction = 0.1)
    X = adata.layers["preprocessed"]
    y = adata.obsm["gating"].toarray()

    X, y = sampler.fit_resample(X, y)

    del sampler
    gc.collect()
    
    if not test_size:
        return X, y

    return (*train_test_split(X,
                              y,
                              test_size = test_size), scaler)