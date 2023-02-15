import json
from typing import *

import pandas as pd
import tensorflow as tf
from matplotlib import rcParams
from model_training_functions import *
from Custom.PlottingFunctions import *
from Custom.TFModels import *
from Custom.TFModelEvaluation import *
from Custom.WindowGenerator import *


if __name__ == "__main__":
    GPU_num = 0
    tf.random.set_seed(42)
    select_GPU(GPU_num)
    # Used EMG features
    features = ["MAV", "RMS"]
    features.extend([f"AR{i+1}" for i in range(6)])
    # Used sensors
    sensors = [f"sensor {i+1}" for i in range(14)] # select sensors numbers (1~14)
    out_labels = ['Right knee angle', 'Right ankle angle', 'Left knee angle','Left ankle angle', 
                  'Right knee moment', 'Right ankle moment','Left knee moment', 'Left ankle moment']
    
    # True if you want to use knee angle as an extra input
    # Labels to be predicted
    out_labels = ['Right knee angle', 'Right ankle angle', 'Left knee angle','Left ankle angle',
                  'Right knee moment', 'Right ankle moment', 'Left knee moment', 'Left ankle moment']
    models_dic = {}

    model_name = "LSTM"
    models_dic = {model_name: create_lstm_model}

    # Create pandas dataframe that will have all the results
    r2_results = pd.DataFrame(columns=out_labels)
    rmse_results = pd.DataFrame(columns=out_labels)
    nrmse_results = pd.DataFrame(columns=out_labels)
    
    subjects = ['01', '05', '06', '07', '10', '13']

    for test_subject in subjects:
        train_subjects = subjects.copy()
        train_subjects.remove(test_subject)
        
        r2, rmse, nrmse, y_true, y_pred = train_fit_gm(
                                        subject=train_subjects,
                                        tested_on=test_subject,
                                        model_name=model_name,
                                        models_dic=models_dic,
                                        epochs=1000,
                                        eval_only=True,
                                        load_best=False,
                                        input_width=20,
                                        shift=1,
                                        label_width=1,
                                        lr=0.003,
                                        batch_size=8,
                                        features=features,
                                        sensors=sensors,
                                        out_labels=out_labels)
        
        plt.close()

        r2_results.loc[int(test_subject), out_labels] = r2
        rmse_results.loc[int(test_subject), out_labels] = rmse
        nrmse_results.loc[int(test_subject), out_labels] = nrmse

    r2_results.to_csv("../Results/intersubject/R2_results.csv")
    rmse_results.to_csv("../Results/intersubject/RMSE_results.csv")
    nrmse_results.to_csv("../Results/intersubject/NRMSE_results.csv")
