"""
Created on Thu Jul 29 14:26:07 2021

@author: amged
"""
# Import libraries
import pandas as pd
import json
import os
import numpy as np
import re
import matplotlib.pyplot as plt


def get_inputs_dir(subject=None):
    if subject == None:
        subject = input("insert subject number: ")
    # Create motion Setting File
    date = subject_details[f"S{subject}"]["date"]
    # Inputs folder path
    input_path = f"../Data/S{subject}/{date}/Dynamics/motion_data/"
    # Get files names (trials)
    trials = ["train_01", "train_02", "val", "test"]
    Inputs = list(map(lambda x: f"S{subject}_{x}.csv", trials))
    # Get inputs and outputs full directories
    Inputs = list(map(lambda file: input_path+file, Inputs))
    return Inputs


def get_markers_labels(Input):
    """
    Input: input data path. Type: string
    """
    # Getting Markers labels
    Markers_Label = pd.read_csv(Input, header=2, nrows=0).columns.values[2:]
    Markers_Label = list(
        map(lambda x: re.sub('\.[0-9]$', "", x), Markers_Label))
    # Do not use set because we do not want to change the order of markers
    unique_labels = []
    for label in Markers_Label:
        if label not in unique_labels:
            unique_labels.append(label)
    unique_labels = list(map(lambda x: re.sub('.+:', "", x), unique_labels))
    return unique_labels


def load_trials_data(Input, Markers_number):
    """
    Input & Output are pathes
    """
    Markers = pd.read_csv(Input, header=5, usecols=[
                          *range(0, 3*Markers_number+2)])
    # After installing the esync, sometimes the sampling rate becomes 100±1 Hz
    Markers['Time (Seconds)'] = [
        i/100 for i in range(len(Markers['Time (Seconds)']))]
    return Markers

def load_v_scarle(subject=None):
    if subject == None:
        subject = input("insert subject number in XX format: ")
    motion_types = ("dynamic")
    flag = True
    for motion_type in motion_types:
        print(f"{motion_type} Data")
        Inputs = get_inputs_dir(subject)
        for Input in Inputs:
            if flag:
                Markers_Label = get_markers_labels(Input)
                flag = False
            markers = load_trials_data(Input, Markers_number=len(Markers_Label))
            markers.set_index('Frame', inplace=True)
            return markers.iloc[:,-3:]
    



if __name__ == '__main__':
    # Load subject details
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)
    data = load_v_scarle('06')
    data['speed'] = 0
    for i in data.index[1:]:
        dx = data.iloc[i,0] - data.iloc[i-1,0]
        dz = data.iloc[i,2] - data.iloc[i-1,2]
        data.iloc[i, -1] = round(np.sqrt(dx**2)*100, 2)

    print('hi')

