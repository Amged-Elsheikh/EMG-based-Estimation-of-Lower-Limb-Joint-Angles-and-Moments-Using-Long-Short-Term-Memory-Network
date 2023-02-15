import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from matplotlib import rcParams

from utilities.model_training_functions import *
from utilities.TFModels import *

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
plt.style.use("ggplot")


if __name__ == "__main__":
    tf.random.set_seed(42) # Select the random number generator to ensure reproducibility of the results
    select_GPU(0) # Select the GPU to be used
    subject = 1 # Select the subjectt for training
    tested_on = None # Select the subjectt for testing. if none will test on train subject
    subject = f"{subject:02d}"
    ######################### Model I/O #########################
    features = ["MAV", "RMS"] # Select features
    features.extend([f"AR{i+1}" for i in range(6)])
    sensors = [f"sensor {i+1}" for i in range(14)] # select sensors numbers (1~14)
    out_labels = ['Right knee angle', 'Right ankle angle', 'Left knee angle','Left ankle angle', 
                  'Right knee moment', 'Right ankle moment','Left knee moment', 'Left ankle moment']
    ###################### Window Parameters ######################
    input_width = 20
    shift = 1
    label_width = 1
    batch_size = 8
    #################### Models names functions ####################
    # If you create new model, just give it a name and pass the function to the dictionary
    model_name = 'LSTM'
    models_dic = {model_name: create_lstm_model}
    ############ Create pd dataframe to hold the results ############
    r2_results = pd.DataFrame(columns=out_labels)
    rmse_results = pd.DataFrame(columns=out_labels)
    nrmse_results = pd.DataFrame(columns=out_labels)
    ################################################
    for subject in range(1, 7):
        subject = f"{subject:02d}"
        
        r2, rmse, nrmse, y_true, y_pred = train_fit(
            subject=subject, # the subject to train the model on
            tested_on=subject, # Subject number, that the model will be evaluated on
            models_dic=models_dic, # Dictionary of all models functions
            model_name=model_name, # Name of the model to be used from the models_dic
            epochs=1000, # Maximum number of epochs to train
            lr=0.003, # learning rate
            eval_only=True, # Do you want to evaluate only (no training). Will load the best model if it exists
            load_best=True, # When training new model, do you want to start from a saved models
            input_width=input_width, # the length of the input time series
            shift=shift, # Output time point distance from thelast input's point on the time series
            label_width=label_width, # How many points you want to predict (set to 1 for now)
            batch_size=batch_size, # The batch size
            features=features, # What features you want to use
            sensors=sensors, # What are the sensors you want to use
            out_labels=out_labels) # Output labels
            
        r2_results.loc[f"S{subject}", out_labels] = r2
        rmse_results.loc[f"S{subject}", out_labels] = rmse
        nrmse_results.loc[f"S{subject}", out_labels] = nrmse
        plt.close()

    r2_results.to_csv("../Results/intrasubject/R2_results.csv")
    rmse_results.to_csv("../Results/intrasubject/RMSE_results.csv")
    nrmse_results.to_csv("../Results/intrasubject/NRMSE_results.csv")

