#! /usr/bin/env python3
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

import matplotlib
# plt.rcParams.update({'font.size': 18})
plt.rc("axes", titlesize=16)
plt.rc("axes", labelsize=16)
plt.rc("xtick", labelsize=14)
plt.rc("ytick", labelsize=14)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False

def plot_csv_files(loc, graph_type="GPU", x_axis_type="num_cosines"):
    print("starting folder: {}".format(loc))
    csv_files = []
    if isinstance(loc, str):
        loc = [loc]
    for loc_i in loc:
        if os.path.isdir(loc_i):
            new_files = glob.glob(loc_i + "/**/*.csv", recursive=True)
            csv_files.extend(new_files)
        elif os.path.isfile(loc_i):
            csv_files.append(loc_i)
        else:
            print("ERROR: Location '{}' does not exist.".format(loc_i))
    df = pd.concat(map(pd.read_csv, csv_files), ignore_index=True, sort=True)
    df.drop_duplicates(inplace=True)
    df = df[["Processor", "GPU", "Num Rollouts", "Method", "Num. Cosines", "Num Timesteps", "Mean Optimization Time (ms)", " Std. Dev. Time (ms)"]]
    if x_axis_type == "num_cosines":
        df = df.loc[df["Num Timesteps"] == 96]
        df = df.sort_values(by=["Num. Cosines"])
    else:
        df = df.loc[df["Num. Cosines"] == 0]
        df = df.sort_values(by=["Num Timesteps"])
    # df = df.sort_values(by=["Num. Cosines"])
    print(df.Method.unique())
    methods = df.Method.unique()
    cpu_names = df.Processor.unique()
    gpu_names = df.GPU.unique()
    gpu_names = [gpu for gpu in gpu_names if not pd.isna(gpu)]
    num_rollouts = df["Num Rollouts"].unique()
    num_rollouts.sort()
    gpu_names.sort()
    methods.sort()
    colors = ["red", "blue", "green", "orange", "cyan", "xkcd:pink", "xkcd:brown", "xkcd:sky blue", "xkcd:magenta"]
    x_axis_data = "Num. Cosines" if x_axis_type == "num_cosines" else "Num Timesteps"
    if graph_type == "gpu":
        for method in methods:
            plt.errorbar(x_axis_data, "Mean Optimization Time (ms)", yerr=" Std. Dev. Time (ms)", data=df.loc[df["Method"] == method])
        plt.legend(labels=methods)
    else:
        for i, gpu in enumerate(gpu_names):
            plt.errorbar("Num Rollouts", "Mean Optimization Time (ms)", yerr=" Std. Dev. Time (ms)",
                         color=colors[i], capsize=2,
                         data=df.loc[(df["Method"] == graph_type) & (df["GPU"] == gpu)])
        plt.legend(labels=gpu_names)
    # plt.xscale("log")
    # plt.yscale("log")
    if x_axis_type == "num_cosines":
        plt.xlabel("Number of Cosines added")
    else:
        plt.xlabel("Number of Timesteps")
    plt.ylabel("Optimization Times [ms]")
    if graph_type == "gpu":
        for gpu_name in gpu_names:
            if not pd.isna(gpu_name):
                gpu_title = gpu_name
                break
        num_rollouts_title = num_rollouts[0]
        if len(num_rollouts) > 1:
            num_rollouts_title = string(num_rollouts_title) + "+"
        plt.title("CPU: {},\nGPU: {},\n {} Samples".format(cpu_names[0], gpu_title, num_rollouts_title))
    else:
        plt.title("{} across GPUs".format(graph_type))
    plt.tight_layout()
    if graph_type == "gpu":
        file_name_type = gpu_names[0]
        file_name_type = file_name_type.replace(" ", "_").lower()
        print(file_name_type)
        plt.savefig("{}_cost_complexity_results.pdf".format(file_name_type), bbox_inches="tight")
    else:
        file_name_type = graph_type
        file_name_type = file_name_type.replace(" ", "_").lower()
        print(file_name_type)
        plt.savefig("{}_cost_complexity_results.pdf".format(file_name_type), bbox_inches="tight")
    # plt.show()

if __name__ == "__main__":
    prefix = os.path.expanduser("~/workspaces/mppi_paper/src/MPPI_Paper_Example_Code/build")
    csv_files = os.getcwd()
    # csv_files = [prefix + "/autorally_sys_compexity_results_2024-03-13_13-32-45.csv"]
    # csv_files = [prefix + "/autorally_sys_compexity_results_2024-03-13_17-55-41.csv"]
    csv_files = [prefix + "/autorally_sys_compexity_results_2024-03-14_10-40-45.csv"]
    # graph_type = "MPPI-Generic"
    # graph_type = "torchrl"
    graph_type = "gpu"
    # x_axis_type = "num_timesteps"
    x_axis_type = "num_cosines"
    plot_csv_files(csv_files, graph_type, x_axis_type)


