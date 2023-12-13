#! /usr/bin/env python3
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def plot_npy_files(file_names):
    all_data = {}
    regex_num_rollouts = "MPPI_(\d+)_"
    search_paths = []
    npy_file_names = []
    if isinstance(file_names, list):
        search_paths = file_names
    elif isinstance(file_names, str):
        search_paths.append(file_names)
    for possible_file in search_paths:
        if os.path.isdir(possible_file):
            npy_files_in_dir = glob.glob(possible_file + "/**/*.npy", recursive=True)
            npy_file_names.extend(npy_files_in_dir)
        else:
            npy_file_names.append(possible_file)

    for file_name in npy_file_names:
        z = re.search(regex_num_rollouts, file_name)
        if z:
            rollout_count = z.group(1)
        else:
            print("no num_rollouts found in name {}".format(file_name))
            exit()

        data = np.load(file_name)
        if "Dynamic Mirror Descent MPPI" in file_name:
            method = "DMD"
        elif "Vanilla MPPI" in file_name:
            method = "MPPI"

        if "control" in file_name:
            data_type = "control"
        elif "state" in file_name:
            data_type = "state"
        elif "cost" in file_name:
            data_type = "cost"
        if method not in all_data:
            dict_entry = {rollout_count: {data_type: data}}
            all_data[method] = dict_entry
        else:
            if rollout_count not in all_data[method]:
                all_data[method][rollout_count] = {data_type: data}
            else:
                all_data[method][rollout_count][data_type] = data
    fig, axes = plt.subplots(3, sharex=True)
    fig.set_tight_layout(True)
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("Accel [$m / s^2$]")
    axes[1].set_ylabel("Position [m]")
    axes[2].set_ylabel("Cost")
    axes[2].set_xlabel("Time")
    colors = ["blue", "xkcd:sky blue", "green", "purple", "red", "orange", "xkcd:lavender"]
    color_choice = 0
    legend_list = []
    for method in all_data.keys():
        for num_rollouts in all_data[method].keys():
            control_data = all_data[method][num_rollouts]["control"][:, 0]
            state_data = all_data[method][num_rollouts]["state"][:, 0]
            cost_data = all_data[method][num_rollouts]["cost"][:, 0]
            label_name = method + " " + num_rollouts
            if num_rollouts == "1024":
                axes[0].plot(range(control_data.shape[0]), control_data, label=label_name, alpha=0.75, color=colors[color_choice])
            axes[2].plot(range(cost_data.shape[0]), cost_data, label=label_name, alpha=0.75, color=colors[color_choice])
            axes[1].plot(range(state_data.shape[0]), state_data, label=label_name, alpha=0.75, color=colors[color_choice])
            print("{} is {}".format(label_name, colors[color_choice]))
            color_choice += 1
            legend_list.append(label_name)

            #     all_data["DMD"]
    plt.legend(legend_list)
    plt.show()

if __name__ == "__main__":
    npy_files = os.getcwd()
    plot_npy_files(npy_files)
