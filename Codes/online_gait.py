from Tringo.TrignoListenerSample import TrignoListener
from Tringo import TrignoClient
from Custom.PlottingFunctions import *
from Custom.TFModels import *
import EMG_Process
from functools import partial
from collections import deque
from sklearn.preprocessing import MinMaxScaler
import time
import numpy as np
import csv
import joblib
from matplotlib.animation import FuncAnimation


class Device:
    # Set Device parameters
    channel_range = (1, 3)  # select the sensor
    features_per_sensor = 2
    features_num = features_per_sensor*channel_range[1]
    time_step = 15
    sample_rate = 1111.1111  # Hz for the sensor
    window_step = EMG_Process.SLIDING_WINDOW_STRIDE
    # new data (unoverlapped) to be added to the query buffer
    samples_per_read = int(window_step * sample_rate)
    window_length = int(EMG_Process.WINDOW_LENGTH * sample_rate)

    apply_np_func = partial(np.apply_along_axis, axis=0)

    def __init__(self, features_functions=[EMG_Process.get_MAV, EMG_Process.wave_length]):
        self.features_functions = features_functions
        # Initialize the device
        self.dev = TrignoClient.TrignoEMG(channel_range=self.channel_range,
                                          samples_per_read=self.samples_per_read,
                                          units="mV", host="localhost")

        self.listener = TrignoListener()
        self.dev.frameListener = self.listener.on_emg

        # Initialize the model

    def run(self):
        self.dev.run()
        print('Waiting for Data..')
        while self.listener.data is None:
            pass

    def stop(self):
        self.dev.stop()

    def collect_new_data(self, emg_deque: deque):
        emg_deque.append(self.listener.data.transpose())

    def get_normalization_parameters(self, measuring_time=10.0, load_scale=False):
        if load_scale:
            try:
                self.scaler = joblib.load('../MinMaxScaler.save')
                self.mean_array = joblib.load('../mean_removal.save')
            except:
                load_scale = False
        if not load_scale:
            # Collect and filter EMG data
            print('START WALKING')
            time.sleep(1)
            filtered_emg = np.apply_along_axis(self.filtering, axis=0,
                                               arr=self.listener.data.transpose())
            # keep filling data for the remaining of the measuring_time period.
            while filtered_emg.shape[0] < measuring_time * 1111.11:
                # Get the current time in seconds
                start_time = time.perf_counter()
                # Make sure not to collect data from small window
                while time.perf_counter() - start_time < self.window_step:
                    pass
                # Get the new window data and filter it
                new_data = np.apply_along_axis(self.filtering, axis=0,
                                               arr=self.listener.data.transpose())
                # Stack New and old data
                filtered_emg = np.vstack((filtered_emg, new_data))
            print('STOP WALKING')
            # remove the mean from the filtered EMG data
            self.mean_array = np.mean(filtered_emg, 0, keepdims=True)
            joblib.dump(self.mean_array, '../mean_removal.save')
            # Get features array from the filtered EMG data
            features = self.get_features_array(filtered_emg)
            # Create the scaler
            self.scaler = MinMaxScaler()
            self.scaler.fit(features)
            # Save the scaler
            joblib.dump(self.scaler, '../MinMaxScaler.save')

    def get_features_array(self, filtered_emg):
        # Sliding window parameters
        start = 0
        end = self.window_length
        step = self.samples_per_read
        # initialize features array.
        features = np.empty(
            shape=(0, self.features_per_sensor*filtered_emg.shape[1]))
        while end < len(filtered_emg):
            window_data = filtered_emg[start: end]
            ##### DEMG
            window_data = self.remove_mean(window_data)
            window_data = np.diff(window_data, axis=0)/0.0009
            window_data = np.array(
                [self.apply_features(window_data[:, i]) for i in range(self.channel_range[1])])
            window_data = window_data.flatten()
            features = np.vstack((features, window_data))
            start += step
            end += step
        return features

    def EMG_2_features(self, emg_deque: deque):
        # Unroll queue to numpy array
        emg = np.empty((0, self.channel_range[1]))
        for i in emg_deque:
            emg = np.vstack((emg, i))

        filtered_emg = self.apply_np_func(self.filtering, arr=emg)
        features = self.get_features_array(filtered_emg)
        features = self.scaler.transform(features)
        return features

    def initialize_DL_model(self, model_func):
        self.model_func = model_func
        self.model = self.model_func(ModelParameters(self.time_step,
                                                     features_num=self.features_num,
                                                     out_nums=1, label_width=1))
        self.model.load_weights(f"../CNN model.hdf5")

    @staticmethod
    def filtering(data):
        """Apply bandpass filter and notch filter to the EMG data"""
        filtered_emg = EMG_Process.notch_filter(
            EMG_Process.bandpass_filter(data))
        return filtered_emg

    def remove_mean(self, data):
        """Remove mean value from all Sensors. The mean array will be calculated only once durning the normalization process"""
        return (data - self.mean_array)

    def apply_features(self, window_data):
        """Extract EMG features from filtered EMG"""
        return np.array([feature_func(window_data)
                         for feature_func in self.features_functions])


class ModelParameters:
    def __init__(self, input_width, features_num, out_nums, label_width):
        self.input_width = input_width
        self.features_num = features_num
        self.out_nums = out_nums
        self.label_width = label_width


if __name__ == "__main__":
    # Collect Normalization data
    tringo_emg = Device()
    tringo_emg.run()
    print("collecting Normalize data will start after 5 seconds")
    time.sleep(0.5)
    print("start collecting normalization data")
    # Get normalization data
    tringo_emg.get_normalization_parameters(measuring_time=5, load_scale=False)
    # Select the prediction model
    tringo_emg.initialize_DL_model(model_func=create_conv_model)
    ######## Initialize EMG Queue
    stride = EMG_Process.SLIDING_WINDOW_STRIDE
    base_window_boxes = EMG_Process.WINDOW_LENGTH/stride
    base_window_points = int(base_window_boxes*Device.samples_per_read)
    number_of_mini_windows = Device.time_step
    number_of_points_in_mini_window = number_of_mini_windows * Device.samples_per_read
    emg_deque = deque(
        maxlen=int(number_of_mini_windows + base_window_boxes))

    # Fill EMG deque with mini-windows
    print("start filling the queue")
    current_time = time.perf_counter()
    while len(emg_deque) < emg_deque.maxlen:
        emg_deque.append(tringo_emg.listener.data.transpose())
        while time.perf_counter() - current_time < tringo_emg.window_step:
            pass

    # Create a csv file to hold the data
    fieldnames = ["time", "ankle moment [N.m/Kg]"]
    with open('live_ankle_data.csv', 'w') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
    t = 0
    # # Start looping after filling the Queue
    Quit = False
    while not Quit:
        try:
            current_time = time.perf_counter()
            # Update emg queue (add to the right, remove from the left)
            tringo_emg.collect_new_data(emg_deque)
            # Extract Features
            features = tringo_emg.EMG_2_features(emg_deque)
            # Reshape the features to fit the model
            features = np.reshape(features, (1, -1, Device.features_num))
            # Apply DL model to the features
            ankle_moment = tringo_emg.model.predict(features)[0][0][0]
            # Update the plot
            with open('live_ankle_data.csv', 'a') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

                info = {
                    "time": t,
                    "ankle moment [N.m/Kg]": ankle_moment
                }

                csv_writer.writerow(info)
                t += stride
                # print(ankle_moment)
            # Avoid overlapping by waiting for the buffer
            while (time.perf_counter() - current_time) < (stride):
                pass
        except KeyboardInterrupt:
            tringo_emg.stop()
            print('Recording stopped manually')
            Quit = True
        except Exception as e:
            tringo_emg.stop()
            Quit = True
            print(e)
