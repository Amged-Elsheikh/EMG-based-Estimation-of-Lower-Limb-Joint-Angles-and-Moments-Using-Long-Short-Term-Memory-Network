from functools import partial
from typing import List
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf


def select_GPU(gpu_index=0):
    '''
    This function will  be used to select which GPU to be used.
    '''
    # Insure CUDA is available
    if not tf.test.is_built_with_cuda():
        raise print("No GPU found")
    else:
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        # Select the GPU
        tf.config.experimental.set_visible_devices(
            devices=gpus[gpu_index], device_type='GPU')
        # Prevent script from booking all available resources. If you removed the followng line, GPU will run on script at the time.
        tf.config.experimental.set_memory_growth(gpus[gpu_index], True)



def create_lstm_model(window_object):
    custom_LSTM = partial(layers.LSTM, units=8, dropout=0.3)
    lstm_model = models.Sequential(
        [
            layers.InputLayer((window_object.input_width,
                              window_object.features_num)),
            custom_LSTM(return_sequences=True),
            custom_LSTM(return_sequences=False),
            layers.Dense(window_object.out_nums * window_object.label_width),
            layers.Reshape(
                [window_object.label_width, window_object.out_nums]),
        ]
    )
    lstm_model.summary()
    return lstm_model


def create_conv_model(window_object):
    conv_model = models.Sequential(
        [
            layers.InputLayer((window_object.input_width,
                              window_object.features_num)),
            layers.Conv1D(filters=16, kernel_size=3,
                          strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.MaxPool1D(pool_size=3, strides=1, padding="valid"),
            layers.LSTM(4, return_sequences=True),
            layers.LSTM(4, return_sequences=False),
            layers.Dense(window_object.out_nums * window_object.label_width),
            layers.Reshape(
                [window_object.label_width, window_object.out_nums])
        ]
    )
    conv_model.summary()
    return conv_model


def create_ff_model(window_object):
    custom_nn = partial(layers.Dense, units=4, activation='selu')
    nn_model = models.Sequential(
        [
            layers.InputLayer((window_object.input_width,
                              window_object.features_num)),
            layers.Flatten(),
            custom_nn(),
            custom_nn(),
            custom_nn(),
            layers.Dense(window_object.out_nums * window_object.label_width),
            layers.Reshape(
                [window_object.label_width, window_object.out_nums]),
        ]
    )
    nn_model.summary()
    return nn_model


def knee_lstm_model(window_object):
    custom_LSTM = partial(layers.LSTM, units=8, dropout=0)
    lstm_model = models.Sequential(
        [
            layers.InputLayer((window_object.input_width,
                              window_object.features_num)),
            custom_LSTM(return_sequences=True),
            custom_LSTM(return_sequences=False),
            layers.Dense(window_object.out_nums * window_object.label_width),
            layers.Reshape([window_object.label_width, window_object.out_nums])]
    )
    lstm_model.summary()
    return lstm_model


def model_callbacks(model_file: str) -> List:
    '''
    Three callback functions are used: Save best model, reducing learning rate and early stop
    '''
    checkpoint_callback = ModelCheckpoint(filepath=model_file,
                                          save_weights_only=True,
                                          monitor="val_loss",
                                          save_best_only=True)

    reduce_lr = ReduceLROnPlateau(monitor="val_loss",
                                  min_delta=1e-3,
                                  factor=0.7,
                                  patience=20)

    early_stop = EarlyStopping(monitor="val_loss",
                               patience=50,
                               restore_best_weights=True)

    return [checkpoint_callback, reduce_lr, early_stop]
