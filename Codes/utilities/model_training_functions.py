from typing import *

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn import metrics

from utilities.DataHandler import DataHandler
from utilities.WindowGenerator import *
from utilities.PlottingFunctions import *
from utilities.TFModels import *

plt.style.use("ggplot")

MUSCLES = {"1": "RF", "2": "VM", "3": "VL", "4": "BF",
           "5": "ST", '6': "TA", '7': "SOL", '8': "GM",
           '9': "PB"}

def train_fit(subject: str, tested_on: Union[str, None], model_name: str, models_dic: Dict,
              epochs=1, lr=0.001, eval_only=False, load_best=False, input_width=20, shift=1,
              label_width=1, batch_size=8, features: List[str] = ['RMS'],
              sensors: List[str] = ['sensor 1'], out_labels: List[str] = [f"ankle moment"]):

    ################################## Get output directories ##################################
    model_file, predictions_pdf, learning_curve_pdf = get_files(subject, is_general_model=False)

    ######### Create Window Object and load trainining and Validation sets #########
    partial_window_generator_func = partial(create_window_generator,
                                            input_width=input_width, shift=shift,
                                            label_width=label_width, batch_size=batch_size,
                                            features=features, sensors=sensors, 
                                            out_labels=out_labels, is_general_model=False)

    window_object = partial_window_generator_func(subject)
    
    train_set, val_set, _ = window_object.make_dataset()

    ################################ Train or Load Model ################################
    model = compile_and_train(window_object, model_name,
                              model_file, models_dic,
                              train_set, val_set,
                              epochs,
                              lr, eval_only, load_best,
                              learning_curve_pdf)

    ################################## Make Estimations ##################################
    if tested_on == None:
        tested_on = subject
    y_true, y_pred = get_estimations(
        tested_on, partial_window_generator_func, model)
    # Rescale the data
    y_true = inverse_scale(y_true, window_object.dataHandler)
    y_pred = inverse_scale(y_pred, window_object.dataHandler)
    
    ############################## Evaluation and plot ##############################
    r2_score, rmse_result, nrmse = evaluation(y_true, y_pred)
    print(f"Subject_{subject} R2: {r2_score}")
    print(f"Subject_{subject} RMSE: {rmse_result}")
    print(f"Subject_{subject} NRMSE: {nrmse}")
    plot_results(y_true, y_pred, out_labels, predictions_pdf)
    
    return r2_score, rmse_result, nrmse, y_true, y_pred


def get_files(subject: str, is_general_model=False) -> List[str]:
    '''
    Get Outputs directories
    '''
    # save Intrasubject and Intersubject models seperatelt
    if is_general_model:
        parent_folder = f"../Results/intersubject/S{subject}"
    else:
        parent_folder = f"../Results/intrasubject/S{subject}"
    # Get the model and plots folder
    model_file = f"{parent_folder}/S{subject} model.hdf5"
    predictions_pdf = f"{parent_folder}/S{subject} predictions.pdf"
    learning_curve_pdf = f"{parent_folder}/S{subject} learning curve.pdf"

    return [model_file, predictions_pdf, learning_curve_pdf]


# # Window generator creation function
def create_window_generator(subject: str, input_width=20, shift=3,
                            label_width=1, batch_size=64, features=["RMS"],
                            sensors=['sensor 1'], out_labels=["ankle moment"],
                            is_general_model=False) -> WindowGenerator:
    '''
    A function what will handle creating the sliding window object
    '''
    try:
        subject = f"{int(subject):02d}"
    except:
        raise 'Subject variable should be a number'
    # Get scaled dataset.
    dataHandler = DataHandler(subject, features, sensors, out_labels)
    # # Create Window object
    window_object = WindowGenerator(dataHandler, input_width,
                                    label_width, shift, batch_size,
                                    is_general_model)
    return window_object


def compile_and_train(window_object: WindowGenerator, model_name: str,
                      model_file: str, models_dic: Dict,
                      train_set: tf.data.Dataset, val_set: tf.data.Dataset,
                      epochs=1, lr=0.003, eval_only=False,
                      load_best=False, learning_curve_pdf: Union[str, None] = None):
    '''This function is responsible for creating and training the model'''
    # Make sure no any cached data are stored
    keras.backend.clear_session()
    # Create the model.
    model = models_dic[model_name](window_object)
    # compile the model using NADAM compiler and a custom Loss function
    model.compile(optimizer=keras.optimizers.Nadam(learning_rate=lr),
                  loss=keras.losses.MeanSquaredError())
    # Set the callbacks
    callbacks = model_callbacks(model_file)
    ############################################################################
    # Loading best model if user specified
    if load_best:
        try:
            model.load_weights(model_file)
        except OSError:
            print("No saved model existing. weights will be initialized")

    try:
        if not eval_only:
            history = model.fit(x=train_set, validation_data=val_set,
                                epochs=epochs, callbacks=callbacks)
            if learning_curve_pdf:
                plot_learning_curve(history, learning_curve_pdf)
            plt.close()

    # To stop the training manually, click Ctrl+C
    except KeyboardInterrupt:
        print("\n\nTrains stopped manually")
    # If there is no trained model to be evaluated, create one

    # except OSError:
    #     print("\n\n No saved model existing. New model will be trained")
    #     eval_only = False
    #     load_best = False
    #     model = compile_and_train(window_object, model_name,
    #                               model_file, models_dic, train_set,
    #                               val_set, epochs, lr, eval_only, 
    #                               load_best, learning_curve_pdf,)
    # Load the best model
    model.load_weights(model_file)
    return model


def get_estimations(tested_on: str, window_generator_func, model: tf.keras.Model) -> Tuple[np.array, np.array]:
    # Create test subject's window object
    window_object: WindowGenerator = window_generator_func(tested_on)
    # Get the evaluation set
    test_set = window_object.evaluation_set
    # Get the predictions
    y_pred = model.predict(test_set)
    # predict will return a 3d array
    if len(y_pred.shape) == 3:
        # Get the last time step and reduce output dimenions to two
        y_pred = y_pred[:, -1, :]
    # Get real outputs from the testset
    for _, y_true in test_set.as_numpy_iterator():
        break
    # make it a 2D vector
    y_true = y_true[:, -1, :]
    return y_true, y_pred


def evaluation(y_true: np.array, y_pred: np.array) -> Tuple[List[Union[float, np.ndarray, Any]]]:
    # Calculate the evaluation metrices
    r2_score = metrics.r2_score(y_true, y_pred, multioutput='raw_values')
    rmse_result = metrics.mean_squared_error(y_true, y_pred, squared=False, multioutput='raw_values')
    nrmse = rmse_result / (np.max(y_true, axis=0) - np.min(y_true, axis=0))
    return np.around(r2_score, 4), np.around(rmse_result, 3), np.around(nrmse, 3)


def inverse_scale(y: np.ndarray, dataHandler: DataHandler):
    angles = dataHandler.angle_scaler.inverse_transform(y[:, :4])
    moment = y[:, 4:] * dataHandler.subject_weight
    return np.column_stack((angles, moment))

def train_fit_gm(subject: List[str], tested_on: str, model_name: str, models_dic: Dict,
                 epochs=1, lr=0.001, eval_only=False, load_best=False, input_width=20, shift=1, 
                 label_width=1, batch_size=8, features: List[str] = ['RMS'],
                 sensors: List[str] = ['sensor 1, '], out_labels: List[str] = [f"ankle moment"]):
    """
    subject: List the subjects used for training.
    tested on: subject number in XX string format.
    """
   ################################## Get Files ##################################
    model_file, predictions_pdf, learning_curve_pdf = get_files(tested_on, is_general_model=True)

    ######### Create Window Object and load trainining and Validation sets ########
    window_generator = partial(create_window_generator,
                               input_width=input_width, shift=shift,
                               label_width=label_width, batch_size=batch_size,
                               features=features, sensors=sensors,
                               out_labels=out_labels, is_general_model=True)
    # Make dataset

    flag = True
    for s in subject:
        if flag:
            flag = False
            window_object = window_generator(s)
            train_set = window_object.train_dataset
            val_set = window_object.val_dataset
        else:
            window_object = window_generator(s)
            temp_train = window_object.train_dataset
            temp_val = window_object.val_dataset

            train_set = train_set.concatenate(temp_train)
            val_set = val_set.concatenate(temp_val)

    train_set = WindowGenerator.preprocessing(train_set, shuffle=True,
                                             batch_size=batch_size, 
                                             drop_reminder=True)

    val_set = WindowGenerator.preprocessing(val_set, shuffle=False,
                                             batch_size=10**6, 
                                             drop_reminder=False)

    ####################### Train or Load Model ###################################
    model = compile_and_train(window_object, model_name,
                              model_file, models_dic,
                              train_set, val_set, epochs,
                              lr, eval_only, load_best,
                              learning_curve_pdf)

    ######################### Make Estimations #####################################
    y_true, y_pred = get_estimations(tested_on, window_generator, model)

    ############################## Evaluation and plot ##############################
    r2_score, rmse_result, nrmse = evaluation(y_true, y_pred)
    print(f"Subject_{tested_on} R2: {r2_score}")
    print(f"Subject_{tested_on} RMSE: {rmse_result}")
    print(f"Subject_{tested_on} NRMSE: {nrmse}")
    
    plot_results(y_true, y_pred, out_labels, predictions_pdf)
    
    return r2_score, rmse_result, nrmse, y_true, y_pred


def add_mean_std(df: pd.Series):
    '''Add mean and Std rows to a pandas sereis'''
    mean = df.mean()
    std = df.std()
    df.loc['mean', :] = mean
    df.loc['std', :] = std


def custom_loss(y_true, y_pred):
    error = tf.reshape(y_true - y_pred, (-1, 1))
    error = error[~tf.math.is_nan(error)]
    return tf.reduce_mean(tf.square(error), axis=0)
