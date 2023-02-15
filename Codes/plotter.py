# from matplotlib import pyplot as plt
# from matplotlib import rcParams
# import json

# from Custom.PlottingFunctions import *

# rcParams['pdf.fonttype'] = 42
# rcParams['ps.fonttype'] = 42
# tick_size = 12
# label_size = 14
# title_size = 20


# def plot_settings(i):
#     if i < 3:
#         plt.ylim([-0.1, 1.51])
#         plt.yticks([0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.5], fontsize=tick_size)
#     else:
#         # plt.ylim([-0.25, 1.52])
#         plt.yticks([0, 0.5, 1, 1.5], fontsize=tick_size)


# with open("subject_details.json", "r") as f:
#     subject_details = json.load(f)

# data = create_window_generator("04").test_df
# # print(data.describe())
# labels = ["Tibialis Anterior", "Gastrocnemius Medialis", 
#                             "Soleus", "Ankle moment"]
# sensors = [f"DEMG{x}_RMS" for x in [1, 3, 5]]
# sensors.append("Left ankle moment")
# data = data.loc[:, sensors]
# mapper = {data.columns[0]: "TA", data.columns[1]: "GM", data.columns[2]: "SOL"}
# data.rename(columns=mapper, inplace=True)

# plt.figure(figsize=(11,9))
# for i, col in enumerate(data.columns):
#     plt.subplot(4, 1, i+1)
#     plt.grid(True)
#     if i == 3:
#         data.loc[:, col] *= -1
#     plt.plot(data.index, data.loc[:, col], linewidth=2.5)
#     plot_settings(i)
#     plt.xticks([i for i in range(0, 10)], fontsize=tick_size)
#     plt.xlim([0, 10])
#     plt.axvspan(0.7, 2.4, alpha=0.3, color='red')
#     plt.axvspan(7.15, 8.75, alpha=0.3, color='red')
#     plt.ylabel("Magnitude")
#     plt.title(labels[i], fontsize=title_size)
# plt.xlabel("time [s]", fontsize=label_size)
# plt.ylabel("N.m/kg", fontsize=label_size)

# plt.tight_layout()
# plt.show()
