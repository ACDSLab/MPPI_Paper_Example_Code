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

def plot_csv_files(loc, graph_type="GPU", compare_across="GPU"):
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
    df = pd.concat(map(pd.read_csv, csv_files), ignore_index=True, sort=True)
    df.drop_duplicates(inplace=True)
    df = df[["Processor", "GPU", "Num Rollouts", "Method", "Mean Optimization Time (ms)", " Std. Dev. Time (ms)"]]
    # torchrl_data = df.loc[df["Method"] == "torchrl"]
    df = df.sort_values(by=["Num Rollouts"])
    methods = df.Method.unique()
    cpu_names = df.Processor.unique()
    gpu_names = df.GPU.unique()
    gpu_names = [gpu for gpu in gpu_names if not pd.isna(gpu)]
    cpu_names.sort()
    cpu_names = np.flip(cpu_names)
    gpu_names.sort()
    methods.sort()
    colors = ["orange", "red", "green", "blue", "cyan", "xkcd:pink", "xkcd:brown", "xkcd:sky blue", "xkcd:magenta"]

    # Create combined items and add both GPU and CPUs to legend
    legends = []
    i = 0
    for cpu in cpu_names:
        for gpu in gpu_names:
            plt.errorbar("Num Rollouts", "Mean Optimization Time (ms)", yerr=" Std. Dev. Time (ms)",
                         color=colors[i], capsize=2,
                         data=df.loc[(df["Method"] == graph_type) & (df["Processor"] == cpu) & (df["GPU"] == gpu)])
            legend_name = cpu + "," + gpu
            legend_name = legend_name.replace("NVIDIA", "")
            legend_name = legend_name.replace(" 6-Core Processor", "")
            legend_name = legend_name.replace("5 ", "")
            legend_name = legend_name.replace("(R) Core(TM)", "")
            legend_name = legend_name.replace("13th Gen ", "")
            legends.append(legend_name)
            i += 1

    plt.legend(labels=legends,fontsize=12)
    plt.xscale("log")
    # plt.yscale("log")
    plt.xlabel("Number of Samples")
    plt.ylabel("Optimization Times [ms]")
    plt.title("{} across Hardware".format(graph_type))
    plt.tight_layout()
    file_name_type = graph_type
    file_name_type = file_name_type.replace(" ", "_").lower()
    file_name_type = file_name_type + "_hw"
    print(file_name_type)
    plt.savefig("{}_results.pdf".format(file_name_type), bbox_inches="tight")

if __name__ == "__main__":
    csv_files = os.getcwd()
    # graph_type = "MPPI-Generic"
    graph_type = "torchrl"
    # graph_type = "gpu"
    compare_across = "Processor"
    # compare_across = "GPU"
    plot_csv_files(csv_files, graph_type, compare_across)


