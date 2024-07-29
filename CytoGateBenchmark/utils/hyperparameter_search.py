import numpy as np
from typing import Literal, Union
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from ..utils._halving_random_search import HalvingRandomSearchCV_TE

def conduct_hyperparameter_search(model,
                                  grid: dict,
                                  method: Literal["HalvingRandomSearchCV",
                                                  "HalvingGridSearchCV",
                                                  "RandomizedSearchCV",
                                                  "GridSearchCV"],
                                  X_train: np.ndarray,
                                  y_train: np.ndarray) -> Union[RandomizedSearchCV, GridSearchCV]:
    n_jobs = 16
    if model.__class__.__name__ == "ExtraTreesClassifier" and y_train.shape[1] == 107:
        n_jobs = 8
        print(f"ExtraTreesClassifier of the human T_cells dataset. Setting n_jobs to {n_jobs}")
    if method == "RandomizedSearchCV":
        total_params = sum(len(grid[key]) for key in grid)
        grid_result = RandomizedSearchCV(estimator = model,
                                         param_distributions = grid,
                                         scoring = "f1_macro",
                                         n_iter = min(total_params, 20),
                                         verbose = 3,
                                         n_jobs = 8,
                                         cv = 5,
                                         error_score = 0.0,
                                         random_state = 187).fit(X_train, y_train)
    elif method == "GridSearchCV":
        grid_result = GridSearchCV(estimator = model,
                                   param_grid = grid,
                                   scoring = "f1_macro",
                                   cv = 5,
                                   n_jobs = 8,
                                   verbose = 3,
                                   error_score = 0.0,
                                   random_state = 187).fit(X_train, y_train)
        
    elif method == "HalvingGridSearchCV":
        raise NotImplementedError("Needs a seperate class")
        #grid_result = HalvingGridSearchCV(estimator = model,
        #                                  param_grid = grid,
        #                                  scoring = "f1_macro",
        #                                  factor = 3,
        #                                  resource = "n_samples",
        #                                  min_resources = 1000,
        #                                  cv = 5,
        #                                  n_jobs = -1,
        #                                  verbose = 3,
        #                                  error_score = 0.0,
        #                                  random_state = 187).fit(X_train, y_train)

    elif method == "HalvingRandomSearchCV":
        grid_result = HalvingRandomSearchCV_TE(estimator = model,
                                               param_distributions = grid,
                                               scoring = "f1_macro",
                                               factor = 3,
                                               resource = "n_samples",
                                               min_resources = 1000,
                                               cv = 5,
                                               n_jobs = n_jobs,
                                               verbose = 3,
                                               error_score = 0.0,
                                               random_state = 187).fit(X_train, y_train)    

    return grid_result
