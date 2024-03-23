#! /usr/bin/env python3
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re

import matplotlib
plt.rc("axes", titlesize=16)
plt.rc("axes", labelsize=16)
plt.rc("xtick", labelsize=14)
plt.rc("ytick", labelsize=14)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False

def plot_csv_files(file_names):
    print("starting folder: {}".format(file_names))
    csv_files = []
    if isinstance(file_names, str):
        file_names = [file_names]
    for loc_i in file_names:
        if os.path.isdir(loc_i):
            new_files = glob.glob(loc_i + "/**/*.csv", recursive=True)
            csv_files.extend(new_files)
        elif os.path.isfile(loc_i):
            csv_files.append(loc_i)
    df = pd.concat(map(pd.read_csv, csv_files), ignore_index=True, sort=True)
    df.drop_duplicates(inplace=True)
    df = df[["Processor", "GPU", "Num Rollouts", "Method", "Mean Optimization Time (ms)", " Std. Dev. Time (ms)",
             "Step Size", "Cost Min", "Cost Mean", "Cost Variance"]]
    df.replace("Dynamic Mirror Descent MPPI", "DMD-MPC", inplace=True)

    # Get relevant column data
    df = df.sort_values(by=["Step Size"])
    step_size_list = df["Step Size"].unique()
    cpus_list = df["Processor"].unique()
    gpus_list = df["GPU"].unique()
    num_rollouts_list = df["Num Rollouts"].unique()
    method_list = df["Method"].unique()
    step_size_list.sort()
    cpus_list.sort()
    gpus_list.sort()
    method_list.sort()
    num_rollouts_list.sort()
    def sqrt_cost_var(row):
        return np.sqrt(row["Cost Variance"])
    df["Cost Std Dev"] = df.apply(sqrt_cost_var, axis=1)

    # Create plots
    num_rows_subplot = 1
    num_cols_subplot = 2
    fig, axes = plt.subplots(num_rows_subplot, num_cols_subplot, figsize=(num_cols_subplot * 5, num_rows_subplot * 5), sharex=True)
    plot_time_over_step_size = True
    # fig, axes = plt.subplots(1, 1)
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    colors = ["red", "blue", "green", "orange", "cyan", "xkcd:pink", "xkcd:brown", "xkcd:sky blue", "xkcd:magenta"]
    plots_list = []
    for method in method_list:
        for num_rollouts in num_rollouts_list:
            plots_list.append((method, num_rollouts))
    for i, (method, num_rollouts) in enumerate(plots_list):
        local_df = df.loc[(df["Method"] == method) & (df["Num Rollouts"] == num_rollouts)]
        cost_mean_min = local_df.min()["Cost Mean"]
        step_size_at_min = local_df.loc[local_df["Cost Mean"].idxmin()]["Step Size"]
        axes[0].scatter(step_size_at_min, cost_mean_min, s=100, color=colors[i], marker="x")
        axes[0].errorbar("Step Size", "Cost Mean", yerr="Cost Std Dev",
                     color=colors[i], capsize=2,
                     data=local_df, label=method + " " + str(num_rollouts) + " Samples")
    # Second plot if needed
    if len(axes) > 1 and not plot_time_over_step_size:
        for method in method_list:
            local_df = df.loc[(df["Method"] == method) & (df["Step Size"] == 1.0)]
            local_df = local_df.sort_values(by=["Num Rollouts"])
            axes[1].errorbar("Num Rollouts", "Mean Optimization Time (ms)", yerr=" Std. Dev. Time (ms)",
                         color=colors[i+2], capsize=2,
                         data=local_df, label="_")
    elif len(axes) > 1 and plot_time_over_step_size:
        for i, (method, num_rollouts) in enumerate(plots_list):
            local_df = df.loc[(df["Method"] == method) & (df["Num Rollouts"] == num_rollouts)]
            axes[1].errorbar("Step Size", "Mean Optimization Time (ms)", yerr=" Std. Dev. Time (ms)",
                         color=colors[i], capsize=2,
                         data=local_df, label=method + " " + str(num_rollouts) + " Samples")
                         # data=local_df, label="_")
    if len(axes) > 1:
        axes[1].legend()
    else:
        axes[0].legend()
    # plt.legend(labels=[method + " " + str(num_rollouts) + " Samples" for method, num_rollouts in plots_list])
    # plt.legend(labels=[str(num_rollouts) + " Samples" for num_rollouts in num_rollouts_list])
    axes[0].set_xlabel("Step Size $\gamma_t$")
    axes[0].set_ylabel("Average Accumulated Cost")
    if len(axes) > 1:
        if plot_time_over_step_size:
            axes[1].set_xlabel("Step Size $\gamma_t$")
        else:
            axes[1].set_xlabel("Number of Samples")
            axes[1].set_xscale("log")
        axes[1].set_ylabel("Optimization Times [ms]")
    # axes[0].set_title("CPU: {},\nGPU: {}".format(cpus_list[0], gpus_list[0]))
    fig.suptitle("CPU: {},\nGPU: {}".format(cpus_list[0], gpus_list[0]), fontsize=16)
    fig.tight_layout()
    file_name_prefix = gpus_list[0]
    file_name_prefix = file_name_prefix.replace(" ", "_").lower()
    fig.savefig("{}_dmd_results.pdf".format(file_name_prefix), bbox_inches="tight")

def plot_npy_files(file_names):
    all_data = {}
    regex_num_rollouts_and_step_size = "MPPI_(\d+)_(\d+\.\d+)_"
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
        z = re.search(regex_num_rollouts_and_step_size, file_name)
        if z:
            rollout_count = z.group(1)
            step_size = z.group(2)
        else:
            z = re.search(regex_num_rollouts, file_name)
            if z:
                rollout_count = z.group(1)
                step_size = None
            else:
                print("no num_rollouts found in name {}".format(file_name))
                exit()

        data = np.load(file_name)
        if "Dynamic Mirror Descent MPPI" in file_name and step_size == 1.0:
            method = "MPPI"
        elif "Dynamic Mirror Descent MPPI" in file_name:
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
    fig, axes = plt.subplots(3, gridspec_kw={"height_ratios": [1, 2, 1]})#, sharex=True)
    fig.set_tight_layout(True)
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("Accel [$m / s^2$]")
    axes[1].set_ylabel("Position [m]")
    axes[1].set(adjustable="box", aspect="equal")
    axes[2].set_ylabel("Cost")
    axes[2].set_xlabel("Time")
    colors = ["blue", "xkcd:sky blue", "green", "purple", "red", "orange", "xkcd:lavender"]
    color_choice = 0
    legend_list = []
    for method in all_data.keys():
        for num_rollouts in all_data[method].keys():
            control_data = all_data[method][num_rollouts]["control"][:, 0]
            state_data = all_data[method][num_rollouts]["state"][:, 0:2]
            cost_data = all_data[method][num_rollouts]["cost"][:, 0]
            label_name = method + " " + num_rollouts
            if num_rollouts == "1024":
                axes[0].plot(range(control_data.shape[0]), control_data, label=label_name, alpha=0.75, color=colors[color_choice])
            axes[2].plot(range(cost_data.shape[0]), cost_data, label=label_name, alpha=0.75, color=colors[color_choice])
            axes[1].plot(state_data[:, 0], state_data[:, 1], label=label_name, alpha=0.75, color=colors[color_choice])
            # axes[1].plot(range(state_data.shape[0]), state_data, label=label_name, alpha=0.75, color=colors[color_choice])
            print("{} is {}".format(label_name, colors[color_choice]))
            color_choice += 1
            legend_list.append(label_name)

            #     all_data["DMD"]
    times = range(state_data.shape[0])
    # goal_pos = [-4 for t in times]
    # axes[1].plot(times, goal_pos, "r--", label="goal")
    # legend_list.append("goal")
    # fig.legend()
    axes[1].set_xlim([-2.25,2.25])
    axes[1].set_ylim([-2.25,2.25])
    plt.legend(legend_list)
    plt.show()

if __name__ == "__main__":
    npy_files = os.getcwd()
    plot_npy_files(npy_files)
    plot_csv_files(os.getcwd())
