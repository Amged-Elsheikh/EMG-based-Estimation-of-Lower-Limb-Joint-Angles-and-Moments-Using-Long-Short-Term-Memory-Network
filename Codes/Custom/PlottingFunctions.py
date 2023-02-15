import matplotlib.pyplot as plt
import pandas as pd
from typing import *
import numpy as np

def plot_learning_curve(history, file: str):
    # Do nothing if train stop manually
    if history == None:
        print("No train history was found")
        return
    else:
        # Create the plot
        plt.figure("Learning curve")
        plt.plot(
            history.epoch,
            history.history["loss"],
            history.epoch,
            history.history["val_loss"])
        
        plt.legend(["train loss", "val loss"])
        plt.xlabel("Epochs")
        plt.ylabel("loss")
        plt.draw()
        # Save according on the desired directories
        plt.savefig(file)
        return

def plot_results(y_true, y_pred, out_labels, file: str):
    tick_size = 12
    label_size = 14
    title_size = 20
    plt.figure("Prediction", figsize=(15, 8))
    time = [i / 20 for i in range(len(y_true))]
    true_df = pd.DataFrame(data=y_true, columns=out_labels, dtype=np.float)
    pred_df = pd.DataFrame(data=y_pred, columns=out_labels, dtype=np.float)
    
    plot_by_side = ['Right knee angle', 'Right ankle angle', 'Left knee angle','Left ankle angle',
                    'Right knee moment', 'Right ankle moment', 'Left knee moment', 'Left ankle moment']
        
    i = 1
    y_label = 'Angle'
    for col in plot_by_side:
        plt.subplot(2, len(out_labels)//2, i)
        
        plt.plot(time, true_df.loc[:, col], "b-", linewidth=2.5)
        plt.plot(time, pred_df.loc[:, col], "r--", linewidth=1.5,)
        
        if i > 4:
            plt.xlabel("Time [s]", fontsize=label_size)
            
        if i == 1 and y_label == 'Angle':
            plt.ylabel("Angle", fontsize=label_size)
            # plt.legend(["Measurments", "Estimations"], fontsize=label_size)
            
        elif i == 5:
            plt.ylabel("Moment [Nm]", fontsize=label_size)
            
        plt.title(col, fontsize=title_size)
        plt.xlim((0, 15))
        # plt.xticks(ticks=[i//5 for i in time if i%5==0],fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        plt.grid(True)
        
        if i==len(out_labels)//2:
            y_label = 'Moment'
        i += 1
    plt.tight_layout()
    plt.savefig(file)
    plt.draw()