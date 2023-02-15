'''
This script is used to convert Motive Motion Capture experement
data (static pose & dynamic trials) to a trc format to be used by OpenSim.
'''
import json
import re
from typing import *

import pandas as pd


def csv2trc(subject: Union[int, str], trials: List[str] = ['test',], motion_types: Union[str, None] = None):
    '''
    This function will handle finding the inputs and outputs directories, 
    process the input data from csv format to trc format which will be used
    by OpenSim software later to build the musculoskeletal model and find 
    joints angles. 
    
    MAKE SURE THE CSV FILE HAVE LABELS MARKERS ONLY
    '''
    try:
        subject = f"{int(subject):02d}"
    except:
        raise 'Subject variable should be a number'
    # If user did not specify the type of motion, then work with both types
    if motion_types == None:
        motion_types = ("static", "dynamic")
    elif type(motion_types) == str:
        motion_types = (motion_types, )

    for motion_type in motion_types:
        print(f"Processing {motion_type} data for subject {subject}")
        # Get inputes and outputs directories
        Inputs, Outputs = get_IO_dir(
            subject, motion_type=motion_type, trials=trials)
        # Loop in trials or process the static pose
        for Input, Output in zip(Inputs, Outputs):
            # Get experement labels once to save run time
            Markers_Label = get_markers_labels(Input)
            # Load markers trajectories
            markers_trajectories = load_trajectories(subject, Input, trials)
            # Prpare the data for OpenSim
            process_trc(markers_trajectories, Output, Markers_Label)


def get_IO_dir(subject: str, motion_type: str = "dynamic",
               trials: List[str] = ['test', ]) -> Tuple[List[str], List[str]]:
    '''
    This function will take the subject number, and the motion type 
    (dynamic or static) to return the inputs and outputs directories.
    '''
    # load experement information
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)
    date = subject_details[f"S{subject}"]["date"]

    # Static pose directories
    if motion_type.lower() == "static":
        # Input file's path
        Inputs = [f"../Data/S{subject}/{date}/Static/S{subject}_static.csv"]
        # Output file's path
        Outputs = [
            f"../Outputs/S{subject}/{date}/Statics/S{subject}_static.trc"]

    # experiment directories
    elif motion_type.lower() == "dynamic":
        # Inputs folder's path
        input_path = f"../Data/S{subject}/{date}/Dynamics/"
        # Outputs folder's path
        output_path = f"../Outputs/S{subject}/{date}/Dynamics/motion_data/"
        # Get files names (trials)
        Inputs = list(map(lambda x: f"S{subject}_{x}.csv", trials))
        Outputs = list(map(lambda x: f"{x}".replace('csv', 'trc'), Inputs))

        # Get inputs and outputs full directories
        Inputs = list(map(lambda file: input_path+file, Inputs))
        Outputs = list(map(lambda file: output_path+file, Outputs))

    else:
        raise f"motion_type should be either 'dynamic' or 'static'\
            (not case sensative). The user input was '{motion_type}'"

    return (Inputs, Outputs)


def get_markers_labels(Input: str) -> List:
    """
    Get the exact markers labels from th directory of any one of the trials
    """
    # Getting markers_trajectories labels
    Markers_Label = pd.read_csv(Input, header=2, nrows=0).columns.values[2:]
    Markers_Label = list(
        map(lambda x: re.sub('\.[0-9]$', "", x), Markers_Label))
    # Do not use set because we do not want to change the order of markers_trajectories
    unique_labels = []
    for label in Markers_Label:
        if label not in unique_labels:
            unique_labels.append(label)
    # Optitrack prefix labels name with the assets name followed by ':'.\
        # The next line will remove the prefix
    unique_labels = list(map(lambda x: re.sub('.+:', "", x), unique_labels))
    return unique_labels


def load_trajectories(subject: str, Input: str, trials: List[str]) -> pd.DataFrame:
    """
    This function will load the Mocap data and trim the trials
    """
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)
    # Read the file
    markers_trajectories = pd.read_csv(Input, header=5)
    # Divide the frame number by the sampling rate to get the time in seconds
    markers_trajectories['Time (Seconds)'] = markers_trajectories["Frame"]/ 100
    # Get trial name
    trial = re.sub(".*S[0-9]*_", "", Input)
    trial = re.sub("\.[a-zA-z]*", "", trial)
    # Trim the data to get the real experement period, but do not do this for the static pose
    if trial in trials:
        record_period = subject_details[f"S{subject}"]["motive_sync"][trial]
        record_start = int(record_period['start'] * 100)
        record_end = int(record_period['end'] * 100)
        # Trim the record but do not change the frame/time values (Used for sync with EMG)
        markers_trajectories = markers_trajectories.iloc[record_start:record_end + 1, :]
    return markers_trajectories


def process_trc(markers_trajectories: pd.DataFrame, Output: str, Markers_Label: List[str]) -> None:
    '''
    This function is used to convert markers data (Pandas/CSV) to OpenSim 
    format`
    '''
    New_label_Coor = '\t'
    New_label_Mar = 'Frame#\tTime'
    Markers_number = len(Markers_Label)
    num_frames = len(markers_trajectories)

    markers_trajectories.to_csv(Output,  sep='\t', index=False, header=False)
    for i in range(0, Markers_number):
        New_label_Coor = f"{New_label_Coor}\tX{str(i+1)}\tY{str(i+1)}\tZ{str(i+1)}"
    for i in range(0, Markers_number-1):
        New_label_Mar = f"{New_label_Mar}\t{Markers_Label[i]}\t\t"
    New_label_Mar = f"{New_label_Mar}\t{Markers_Label[Markers_number-1]}\n"

    Contents = 'PathFileType\t4\t' + '(X,Y,Z)\t' + Output + '\n' \
        + 'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames' + '\n' \
        + f'100\t100\t{num_frames}\t{Markers_number}\tm\t100\t1\t{num_frames}\n'

    with open(Output, "r+") as f:
        old = f.read()  # read everything in the file
        f.seek(0)  # rewind
        f.write(Contents + New_label_Mar + New_label_Coor + '\n\n' + old)


if __name__ == '__main__':
    subject = input('Please write the subject number: ')
    trials = ["trial"]
    # dynamic trials to be processed
    csv2trc(subject=subject, trials=trials, motion_types=None)
