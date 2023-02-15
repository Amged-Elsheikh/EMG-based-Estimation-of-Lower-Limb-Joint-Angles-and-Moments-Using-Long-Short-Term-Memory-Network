"""
This script will process the data from force plate 1 only. 
"""
import json
import os
import re
from typing import *

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

pd.set_option("mode.chained_assignment", None)


def GRF_to_OpenSim(
    subject: Union[str, int],
    trials: List[str] = [
        "test",
    ],
    offset_remove: bool = False,
) -> None:
    """
    This function will handle finding the inputs and outputs directories,
    process the input data from csv format to sto format which will be used
    by OpenSim software later to solve the inverse dynamic equation.
    """
    try:
        subject = f"{int(subject):02d}"
    except:
        raise "Subject variable should be a number"
    # Get the inputs and outputs directories
    input_path, output_path, files = get_IO_dir(subject, trials)
    # Process each trial
    data = dict()

    for i, file in enumerate(files):
        # Load Left force plates data
        for side_file in file:
            if "forceplate_1" in side_file:
                side = "L"
            elif "forceplate_2" in side_file:
                side = "R"
            else:
                raise
            data[side] = pd.read_csv(
                input_path + side_file, header=31, low_memory=False
            )

            # System sometimes stop sending data for few frames
            data[side] = remove_system_gap(data[side])

            # Remove the delay
            data[side] = shift_data(data[side], subject, shift_key=trials[i])

            # Remove the offset from the data
            if remove_offset:
                data[side] = remove_offset(data[side], remove=offset_remove)
                
            # Crop the data to get the actual experement
            data[side] = trial_period(data[side], subject, trials[i])

            data[side] = fix_CoP(data[side])

            # Match devices coordinate system
            data[side] = system_match(data[side], side)

        # Rename columns to match OpenSim default names
        force_data = GRF_data(data)
        # Save force data
        output_name = re.sub("_forceplate_[0-9].csv", "_grf.sto", side_file)
        save_force_data(force_data, output_path, output_name)


def get_IO_dir(subject: str, trials: List[str]) -> Tuple[str, str, Tuple[List[str]]]:
    """
    This fynction will generate inputs and outputs directories for the selected\
    subject and trials.
    """
    # Load experiment information
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)

    date = subject_details[f"S{subject}"]["date"]
    input_path = f"../Data/S{subject}/{date}/Dynamics/"
    output_path = f"../Outputs/S{subject}/{date}/Dynamics/Force_Data/"
    files = [
        (f"S{subject}_{trial}_forceplate_1.csv", f"S{subject}_{trial}_forceplate_2.csv")
        for trial in trials
    ]
    return input_path, output_path, files

    # # Match devices coordinate system
    # data = system_match(data)


def trial_period(data: pd.DataFrame, subject: str, trial: str) -> pd.DataFrame:
    """
    this function will trim the experiment to the actual experiment period and\
    create the time column
    """
    # Load experiment information
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)
    # get the actual experiment period and force plates sampling rate
    record_period = subject_details[f"S{subject}"]["motive_sync"][trial]
    fps = 100
    # Create time column by dividing frame number by the sampling rate
    data["time"] = data[" DeviceFrame"] / fps
    # Trim the experiment to the actual experiment period
    record_start = int(record_period["start"] * fps)
    record_end = int(record_period["end"] * fps)
    data = data.iloc[record_start : record_end + 1, :]
    # Reset the index column but keep the time column
    data.reset_index(inplace=True, drop=True)
    return data


def remove_system_gap(data: pd.DataFrame) -> pd.DataFrame:
    """
    In some cases force plates stop recording and send zeros.it's not common to have
    exactly 0, usually the reading will be small float number
    This function will first set these values to NaN and then perform linear interpolation.
    """
    # Get the main sensors' data from the dataset. CoP is calculated based on the force and moment
    # not measured by a sensor directly.
    columns = [" Fx", " Fy", " Fz", " Mx", " My", " Mz"]
    # Zero means no data sent in almost all cases
    data.loc[data.loc[:, " Fz"] == 0, columns] = np.nan
    data.iloc[:, :] = data.interpolate(method="linear")
    data.iloc[:, :] = data.fillna(method="bfill")
    return data


def shift_data(data: pd.DataFrame, subject, shift_key) -> pd.DataFrame:
    """
    In early experements there was no external synchronization between the Mocap
    and force plates, resulting in a starting delay. This delay is different every
    time the start button is pressed to start new experiment/trial.

    This function will shift the data by a number of frames specified by the user
    in the experiment json file with 'delay' as a key value.
    """
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)

    shift_value = subject_details[f"S{subject}"]["delay"][shift_key]
    if shift_value != 0:
        shift_columns = [" Fx", " Fz", " Fy", " Mx", " Mz", " My", " Cx", " Cz", " Cy"]
        data.loc[:, shift_columns] = data[shift_columns].shift(
            shift_value, fill_value=0
        )
    return data


def remove_offset(data: pd.DataFrame, remove: bool = True) -> pd.DataFrame:
    """
    Force plate sensors have a small amount of offset. It can be removed bt
    finding the average offset value and substracting from the dataset if the
    user want to do so.
    """
    if remove:
        # Choose Forces and Moments
        columns = [" Fx", " Fy", " Fz", " Mx", " My", " Mz"]
        for col in columns:
            data.loc[:, col] = data.loc[:, col] - data.loc[5:15, col].mean()
    return data


def fix_CoP(data: pd.DataFrame) -> pd.DataFrame:
    """
    This function is responsible for applying a Butterworth lowpass filter
    for the force and moment data when the subject is on the force plate.
    The filter low frequency will be adjusted automatically depending on
    the sample rate of the force plate.

    The function will detect when the subject is on the force plate if
    the Fz value is greater the 10% of the subject's weight.

    """
    # Recalculate the CoP
    data[" Cx"] = -data[" My"] / data[" Fz"]
    data[" Cy"] = data[" Mx"] / data[" Fz"]
    # If the CoP is outside the force plate, put it at it's center
    data[" Cx"].loc[abs(data[" Cx"]) > 0.25] = 0
    data[" Cy"].loc[abs(data[" Cy"]) > 0.25] = 0
    # apply the filter
    return data


def system_match(data: pd.DataFrame, side: str) -> pd.DataFrame:
    """
    This funchion will match opti-track and force plates axes,
    so that OpenSim can correctly solve the inverse dynamic equation.
    """
    # To apply rotation, change column names.
    col_names = {
        " Fx": " Fx",
        " Fy": " Fz",
        " Fz": " Fy",
        " Mx": " Mx",
        " My": " Mz",
        " Mz": " My",
        " Cx": " Cx",
        " Cy": " Cz",
        " Cz": " Cy",
    }
    data.rename(columns=col_names, inplace=True)
    # Complete the rotation
    change_sign = [" Fx", " Fz"]
    data.loc[:, change_sign] = -data.loc[:, change_sign]
    # Match opti-track and force Plates origins
    if side == 'L':
        data.loc[:, " Cx"] = data[" Cx"] + 0.25
    elif side == "R":
        data.loc[:, " Cx"] = data[" Cx"] + 0.75
    else:
        raise
    data.loc[:, " Cz"] = data[" Cz"] + 0.25
    return data


def GRF_data(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Format the columns name to match the names in OpenSim tutorials.
    It's possible to skip this phase, but I won't recommend it at all.
    """
    # make sure columns are well arranged to save time when working with OpenSim.
    for side in data_dict.keys():
        data_dict[side] = data_dict[side][["time", " Fx", " Fy", " Fz", " Mx", " My", " Mz", " Cx", " Cy", " Cz"]]
        # Rename the columns
        columns_names_mapper = {
            " Fx": f"{side}_ground_force_vx",
            " Fy": f"{side}_ground_force_vy",
            " Fz": f"{side}_ground_force_vz",
            " Cx": f"{side}_ground_force_px",
            " Cy": f"{side}_ground_force_py",
            " Cz": f"{side}_ground_force_pz",
            " Mx": f"{side}_ground_torque_x",
            " My": f"{side}_ground_torque_y",
            " Mz": f"{side}_ground_torque_z",
        }
        data_dict[side].rename(columns=columns_names_mapper, inplace=True)
        
    if len(data_dict.keys()) == 2:
        data = data_dict['L'].merge(data_dict['R'], on='time')
        return data
    elif len(data_dict.keys()) == 1:
        return data_dict[side]
    else:
        raise


def save_force_data(
    force_data: pd.DataFrame, output_path: str, output_name: str
) -> None:
    """
    This function is used to convert processed force plate data (Pandas/CSV) to OpenSim
    format.
    """
    output_file = output_path + output_name
    if os.path.exists(output_file):
        os.remove(output_file)
    force_data.to_csv(output_file, sep="\t", index=False)
    nRows = len(force_data)  # end_time - start_time + 1
    nColumns = len(force_data.columns)

    with open(output_file, "r+") as f:
        old = f.read()  # read everything in the file
        f.seek(0)  # rewind
        f.write(
            output_name
            + "\n"
            + "version=1\n"
            + f"nRows={nRows}\n"
            + f"nColumns={nColumns}\n"
            + "inDegrees=yes\n"
            + "endheader\n"
            + old
        )


if __name__ == "__main__":
    # subject = input('Please write the subject number: ')
    trials = ["trial"]
    for subject in [1, 4, 5, 7, 10, 13]:
        print(f"Subject {subject}")
        GRF_to_OpenSim(
            subject=subject,
            trials=trials,
            offset_remove=True,
        )
