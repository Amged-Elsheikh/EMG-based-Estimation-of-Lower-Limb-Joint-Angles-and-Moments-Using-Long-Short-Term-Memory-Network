import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, filtfilt
from typing import *


class DataHandler:
    '''
    A custom class that will make working with data easier.
    '''
    def __init__(self, subject: str, features: List[str], sensors: List[str],
                 out_labels: List[str] = ["Left ankle moment"]):

        # Initiate the subject
        self.subject = subject
        self.features = features
        self.sensors = sensors
        self.out_labels = out_labels
        
        #Load data and store it in a dictionary. The self method with the dictionary will allow accessing the dataset easily.
        self._data = pd.read_csv(f"../Dataset/S{subject}_trial_dataset.csv",
                                index_col="time")
        
        self._labels_lowpass_filter()
        
        # Create a list of features columns
        self.emg_features = []
        # Get the desired EMG features
        for sensor in self.sensors:
            for feature in self.features:
                self.emg_features.append(f'{sensor} {feature}')
                
        self.angle_cols = list(filter(lambda x: 'angle' in x, self.out_labels))
        self.moment_cols = list(filter(lambda x: 'moment' in x, self.out_labels))
        
        
        # Count the number of features
        self.features_num = len(self.emg_features)
        # initiate the columns for the models (I&O)
        self.model_columns = self.emg_features.copy()
        # Add the output columns for the model columns (Columns used for the model)
        self.model_columns.extend(self.out_labels)
        # Scale the data
        self._joints_columns = list(
            filter(lambda x: "sensor" not in x, self._data.columns))
        
        # Divide data into train val test
        self.train_set = self._data.iloc[: int(0.6 * len(self._data)), :]
        self.val_set = self._data.iloc[int(0.6 * len(self._data)): int(0.8 * len(self._data)), :]
        self.test_set = self._data.iloc[int(0.8 * len(self._data)): , :]
        
        self._is_scaler_available = False

        self.train_set = self._scale(self.train_set)
        self.train_set = self.train_set.loc[:, self.model_columns]
        
        self.val_set = self._scale(self.val_set)
        self.val_set = self.val_set.loc[:, self.model_columns]
        
        self.test_set = self._scale(self.test_set)
        self.test_set = self.test_set.loc[:, self.model_columns]

    @ property
    def subject_weight(self) -> float:
        with open("subject_details.json", "r") as f:
            return json.load(f)[f"S{self.subject}"]["weight"]

    def _scale(self, data):
        '''
        Scale the Dataset
        '''
        # Scale features between 0 and 1
        if not self._is_scaler_available:
            self._features_scaler = MinMaxScaler(feature_range=(0, 1))
            # The scaler will fit only data from the recording periods.
            self._features_scaler.fit(data.loc[:, self.emg_features])
        data.loc[:, self.emg_features] = self._features_scaler.transform(
                                                data.loc[:, self.emg_features])

        # scale angles
        if not self._is_scaler_available:
            self.angle_scaler = MinMaxScaler(feature_range=(0, 1))
            self.angle_scaler.fit(data.loc[:, self.angle_cols])
        data.loc[:, self.angle_cols] = self.angle_scaler.transform(
                                                    data.loc[:, self.angle_cols])
        
        # Scale moments by subjext's weight
        data.loc[:, self.moment_cols] = data.loc[:, self.moment_cols] / self.subject_weight
        # Set the scaler value to True to avoid creating new scalers
        self._is_scaler_available = True
        return data
    
    def _labels_lowpass_filter(self, freq=6, fs=20):
        low_pass = freq / (fs / 2)
        b2, a2 = butter(N=6, Wn=low_pass, btype="lowpass")
        # Don't filter the time column
        self._data.iloc[:, -8:] = filtfilt(b2, a2, self._data.iloc[:, -8:], axis=0)
