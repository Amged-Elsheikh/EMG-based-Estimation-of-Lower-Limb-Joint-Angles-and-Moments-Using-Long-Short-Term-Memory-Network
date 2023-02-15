'''
Create all necessary directories in a proper format.
'''
import json
import os

subject = input("Please write subject number in XX format: ")
subject = f"{int(subject):02d}"

with open("subject_details.json", "r") as f:
    subject_details = json.load(f)[f"S{subject}"]
    
date = subject_details["date"]

setup_subs = ["GRF", "ID", "IK", "Scale"]
setup_subs = list(map(lambda x: f"{x}_setups", setup_subs))

outputs = {"motions_folder": f"Outputs/S{subject}/{date}/Dynamics/Motion_Data",
            "forces_folder": f"Outputs/S{subject}/{date}/Dynamics/Force_Data",
            "static_folder": f"Outputs/S{subject}/{date}/Statics",
            "DEMG": f"Outputs/S{subject}/{date}/DEMG",
            "dataset": f"Dataset/S{subject}",
            "IK": f"Outputs/S{subject}/{date}/IK",
            "Model": f"Outputs/S{subject}/{date}/Model",
            "ID": f"Outputs/S{subject}/{date}/trial/ID",
            "setups": f"Outputs/S{subject}/{date}/Setups"}

# Create Outputs folder (to be filled from/after running script)

for folder in outputs.keys():
    if folder == "setups":
        for sub in setup_subs:
            if not os.path.exists(f"../{outputs[folder]}/{sub}"):
                os.makedirs(f"../{outputs[folder]}/{sub}")
    else:
        if not os.path.exists(f"../{outputs[folder]}"):
            os.makedirs(f"../{outputs[folder]}")
