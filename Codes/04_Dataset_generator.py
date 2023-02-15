"""
This code will create subject's dataset. the dataset will contains: EMG features, Joints kinematics and kinetics.
"""
import json
from typing import *

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def get_directories(subject: int, trials) -> List[List[str]]:
    """
    Load all experiment data directories and the dataset directories
    """
    # Load the subject details
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)
    # Get IK directories
    base_dir = f"../Outputs/S{subject}"

    ik_path = f"{base_dir}/IK"
    IK_files = list(map(lambda x: f"{ik_path}/S{subject}_IK_{x}.mot", trials))

    # Get ID directories
    id_path = f"{base_dir}/ID"
    ID_files = list(map(lambda x: f"{id_path}/{x}/inverse_dynamics.sto", trials))
    # Get the record's periods directories

    # Get the EMG features directories
    features_path = f"{base_dir}/DEMG"
    Features_files = list(map(lambda x: f"{features_path}/{x}_features.csv", trials))
    # Get Dataset directories
    output_files = list(map(lambda x: f"../Dataset/S{subject}_{x}_dataset.csv", trials))
    return [IK_files, ID_files, Features_files, output_files]


def load_data(ik_file: str, id_file: str, features_file: str) -> List[pd.DataFrame]:
    """
    This function will load the trial data
    """
    # Load IK data
    IK = pd.read_csv(ik_file, header=8, sep="\t", usecols=[0, 10, 11, 17, 18])
    # Load ID data
    ID = pd.read_csv(id_file, header=6, sep="\t", usecols=[0, 16, 18, 17, 19])
    # Load EMG features
    features = pd.read_csv(features_file, index_col="time")
    return [IK, ID, features]


def merge_joints(IK: pd.DataFrame, ID: pd.DataFrame) -> pd.DataFrame:
    """Join kinematics and kinetics data into a single pandas dataframe along with record periods
    dataframe which used to remove the moment values (set as NaN) outside the recording period (space)
    """
    # Filter the moments
    ID.iloc[:, :] = lowpass_filter(ID)
    # Merge kinematics and kinetics data on the time column
    joints_data = pd.merge(IK, ID, on="time", how="inner")
    # Reset time to zero to match EMG
    joints_data = reset_time(joints_data)
    return joints_data


def lowpass_filter(data: pd.DataFrame, freq=5, fs=100):
    low_pass = freq / (fs / 2)
    b2, a2 = butter(N=6, Wn=low_pass, btype="lowpass")
    # Don't filter the time column
    data.iloc[:, 1:] = filtfilt(b2, a2, data.iloc[:, 1:], axis=0)
    return data


def reset_time(data: pd.DataFrame) -> pd.DataFrame:
    """
    This function will reset the time from zero by removing the minimum value
    """
    # start_time = data['time'].min()
    # data['time'] = data['time'].apply(lambda x: x-start_time)
    data["time"] = data["time"] - data["time"].min()
    # Round the time into 3 digits (very important because of python limitations)
    data["time"] = np.around(data["time"], 3)
    return data


def merge_IO(features: pd.DataFrame, joints_data: pd.DataFrame) -> pd.DataFrame:
    """
    This function will create the dataset by merging the inputs and outputs.
    IK and ID data downsampling is hold while merging by removing time point
    that's are not shared between the features and joints data
    """
    # Merge all features and joints. Down sampling is done while merging using 'inner' argument
    Dataset = pd.merge(left=features, right=joints_data, on="time", how="inner")
    Dataset.set_index("time", inplace=True)
    return Dataset


def get_dataset(subject: Union[str, int], trials=["trial"]):
    try:
        subject = f"{int(subject):02d}"
    except:
        raise "Subject variable should be a number"
    ########################### Get I/O directories ###########################
    IK_files, ID_files, Features_files, output_files = get_directories(subject, trials)
    ########################### Loop in each trial ###########################
    for trial in range(len(trials)):
        # Load the data
        IK, ID, features = load_data(
            IK_files[trial], ID_files[trial], Features_files[trial]
        )
        # Merge IK, ID & record intervals together to create joint's dataset
        joints_data = merge_joints(IK, ID)
        # Merge EMG features with joints data and down sample joints data to match the EMG features
        Dataset = merge_IO(features, joints_data)
        # Rename column and save the dataset
        new_col = {
            "knee_angle_l": "Left knee angle",
            "ankle_angle_l": "Left ankle angle",
            "ankle_angle_l_moment": "Left ankle moment",
            "knee_angle_l_moment": "Left knee moment",
            "knee_angle_r": "Right knee angle",
            "ankle_angle_r": "Right ankle angle",
            "ankle_angle_r_moment": "Right ankle moment",
            "knee_angle_r_moment": "Right knee moment",
        }
        Dataset.rename(columns=new_col, inplace=True)
        # Save the dataset
        Dataset.to_csv(output_files[trial])


if __name__ == "__main__":
    # subject = input('Please write the subject number: ')
    for subject in [1,]:
        get_dataset(subject=subject, trials=["trial"],)
