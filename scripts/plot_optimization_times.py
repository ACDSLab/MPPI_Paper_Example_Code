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

def plot_csv_files(loc, graph_type="GPU"):
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
    print(df.Method.unique())
    methods = df.Method.unique()
    cpu_names = df.Processor.unique()
    gpu_names = df.GPU.unique()
    gpu_names = [gpu for gpu in gpu_names if not pd.isna(gpu)]
    gpu_names.sort()
    colors = ["red", "blue", "green", "orange", "cyan", "xkcd:pink", "xkcd:brown", "xkcd:sky blue", "xkcd:magenta"]
    if graph_type == "gpu":
        for method in methods:
            plt.errorbar("Num Rollouts", "Mean Optimization Time (ms)", yerr=" Std. Dev. Time (ms)", data=df.loc[df["Method"] == method])
        plt.legend(labels=methods)
    else:
        for i, gpu in enumerate(gpu_names):
            plt.errorbar("Num Rollouts", "Mean Optimization Time (ms)", yerr=" Std. Dev. Time (ms)",
                         color=colors[i], capsize=2,
                         data=df.loc[(df["Method"] == graph_type) & (df["GPU"] == gpu)])
        plt.legend(labels=gpu_names)
    # plt.bar("Num Rollouts", "Mean Optimization Time (ms)", yerr=" Std. Dev. Time (ms)", data=df.loc[df["Method"] == "ros2"])
    # plt.bar("Num Rollouts", "Mean Optimization Time (ms)", data=torchrl_data)
    # plt.errorbar("Num Rollouts", "Mean Optimization Time (ms)", yerr=" Std. Dev. Time (ms)", data=df.loc[df["Method"] == "ros2"])
    # plt.errorbar("Num Rollouts", "Mean Optimization Time (ms)", yerr=" Std. Dev. Time (ms)", data=torchrl_data)
    # plt.errorbar("Num Rollouts", "Mean Optimization Time (ms)", yerr=" Std. Dev. Time (ms)", data=df.loc[df["Method"] == "MPPI-Generic"])
    # plt.legend(labels=["ros2", "torchrl", "MPPI-Generic"])
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of Samples")
    plt.ylabel("Optimization Times [ms]")
    if graph_type == "gpu":
        for gpu_name in gpu_names:
            if not pd.isna(gpu_name):
                gpu_title = gpu_name
                break
        plt.title("CPU: {},\nGPU: {}".format(cpu_names[0], gpu_title))
    else:
        plt.title("{} across GPUs".format(graph_type))
    plt.tight_layout()
    # plt.legend(df["Method"].values)
    # ax.legend(title="Method")
    if graph_type == "gpu":
        plt.savefig("test_mppi_approach_plot.pdf", bbox_inches="tight")
    else:
        file_name_type = graph_type
        file_name_type = file_name_type.replace(" ", "_").lower()
        print(file_name_type)
        plt.savefig("{}_results.pdf".format(file_name_type), bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    prefix = os.path.expanduser("~/workspaces/mppi_workspace/")
    csv_files = os.getcwd()
    # graph_type = "MPPI-Generic"
    # graph_type = "torchrl"
    graph_type = "gpu"
    plot_csv_files(csv_files, graph_type)


