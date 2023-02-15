from functools import partial
from typing import *

import numpy as np
from sklearn import metrics


def nan_R2(y_true: np.array, y_pred: np.array) -> List[Union[float, np.ndarray, Any]]:
    '''
    Get R2 score. This function is customed to handle the case of having missing values
    '''
    R2 = []
    # When having multiple joints (outputs), it's better to have different R2 for each
    _, joints_num = np.shape(y_true)
    for joint in range(joints_num):
        # Get joint true and estimated data
        y_true_col = y_true[:, joint]
        y_pred_col = y_pred[:, joint]
        # Create a logic to remove Nan
        logic = np.isfinite(y_true_col)
        y_true_col = y_true_col[logic]
        y_pred_col = y_pred_col[logic]
        # Calculate and append the R2
        R2.append(metrics.r2_score(y_true_col, y_pred_col))
    return np.around(R2, 4)


def nan_rmse(y_true: np.array, y_pred: np.array) -> Tuple[List[Union[float, np.ndarray, Any]]]:
    RMSE = partial(metrics.mean_squared_error, squared=False)
    rmse_error = []
    nrmse = []
    # When having multiple joints (outputs), it's better to have different R2 for each
    _, joints_num = np.shape(y_true)
    for joint in range(joints_num):
        # Get joint true and estimated data
        y_true_col = y_true[:, joint]
        y_pred_col = y_pred[:, joint]
        # Create a logic to remove Nan
        logic = np.isfinite(y_true_col)
        y_true_col = y_true_col[logic]
        y_pred_col = y_pred_col[logic]
        # Calculate and append the RMSE and the normalized RMSE (Normalize by MinMAX of y_true)
        rmse_error.append(RMSE(y_true_col, y_pred_col))
        delta = max(y_true_col) - min(y_true_col)
        nrmse.append(rmse_error[joint]/delta)
    return np.around(rmse_error, 3), np.around(nrmse,3)
