#! /usr/bin/env python3
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False

def plot_csv_files(loc):
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
    df = pd.concat(map(pd.read_csv, csv_files), ignore_index=True)
    print(df)
    for col in df:
        print(col)

    # torchrl_data = df.loc[df["Method"] == "torchrl"]
    df = df.sort_values(by=["Num Rollouts"])
    print(df.Method.unique())
    methods = df.Method.unique()
    cpu_names = df.Processor.unique()
    gpu_names = df.GPU.unique()
    print(cpu_names)
    print(gpu_names)
    for method in methods:
        plt.errorbar("Num Rollouts", "Mean Optimization Time (ms)", yerr=" Std. Dev. Time (ms)", data=df.loc[df["Method"] == method])
    # plt.bar("Num Rollouts", "Mean Optimization Time (ms)", yerr=" Std. Dev. Time (ms)", data=df.loc[df["Method"] == "ros2"])
    # plt.bar("Num Rollouts", "Mean Optimization Time (ms)", data=torchrl_data)
    # plt.errorbar("Num Rollouts", "Mean Optimization Time (ms)", yerr=" Std. Dev. Time (ms)", data=df.loc[df["Method"] == "ros2"])
    # plt.errorbar("Num Rollouts", "Mean Optimization Time (ms)", yerr=" Std. Dev. Time (ms)", data=torchrl_data)
    # plt.errorbar("Num Rollouts", "Mean Optimization Time (ms)", yerr=" Std. Dev. Time (ms)", data=df.loc[df["Method"] == "MPPI-Generic"])
    # plt.legend(labels=["ros2", "torchrl", "MPPI-Generic"])
    plt.legend(labels=methods)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of Rollouts")
    plt.ylabel("Optimization Times [ms]")
    plt.title("Results for CPU: {}, GPU: {}".format(cpu_names[0], gpu_names[0]))
    plt.tight_layout()
    # plt.legend(df["Method"].values)
    # ax.legend(title="Method")
    plt.savefig("test_mppi_approach_plot.pdf", bbox_inches="tight")
    plt.show()

    # for csv_i in csv_files:
    #     pd.read_csv(csv_i)


if __name__ == "__main__":
    prefix = os.path.expanduser("~/workspaces/mppi_workspace/")
    csv_files = os.getcwd()
    plot_csv_files(csv_files)


