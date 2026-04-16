from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DO_ANNOTATE = False

RESULTS_DIR = Path(__file__).parent / "results"

# get a color palette
dfx_adaptive_colors = sns.color_palette("Blues", 6)
dfx_fixed_colors = sns.color_palette("Greens", 2)
mjx_fixed_colors = sns.color_palette("Reds", 3)

settings_dct = OrderedDict(
    [
        ("mjx_euler", (mjx_fixed_colors[0], "MJX (Euler-SemiImplicit)", "<")),
        ("mjx_rk4", (mjx_fixed_colors[1], "MJX (RK4)", ">")),
        ("mjx_implicit", (mjx_fixed_colors[2], "MJX (ImplicitFast)", "o")),
        #
        ("dfx_euler", (dfx_fixed_colors[0], "DiffMJX (Euler)", "s")),
        ("dfx_eulersemi", (dfx_fixed_colors[1], "DiffMJX (Euler-SemiImplicit)", "D")),
        #
        ("dfx_heun", (dfx_adaptive_colors[0], "DiffMJX* (Heun2)", "P")),
        ("dfx_bosh3", (dfx_adaptive_colors[1], "DiffMJX* (Bosh3)", "H")),
        ("dfx_dopri5", (dfx_adaptive_colors[2], "DiffMJX* (Dopri5)", "v")),
        ("dfx_tsit5", (dfx_adaptive_colors[3], "DiffMJX* (Tsit5) (recommended)", "X")),
        ("dfx_tsit5approx", (dfx_adaptive_colors[4], "DiffMJX* (Tsit5-Approx)", "<")),
        ("dfx_dopri8", (dfx_adaptive_colors[5], "DiffMJX* (Dopri8)", "x")),
        ("dfx_tsit5backsolve", ((0, 0, 0), "DiffMJX* (Tsit5-Backsolve)", ">")),
    ]
)


def get_group(name):
    for prefix in sorted(settings_dct.keys(), key=len, reverse=True):
        if name.startswith(prefix):
            return prefix, name[len(prefix) + 1 :]
    return name, ""


def main():
    df = pd.read_csv(RESULTS_DIR / "cube_results.csv", index_col=0)
    df = df.sort_index()

    settings_list = df.index.tolist()
    groups = []
    specs = []
    for setting in settings_list:
        group, spec = get_group(setting)
        groups.append(group)
        specs.append(spec)
    df["group"] = groups
    df["spec"] = specs

    groups_unique = list(settings_dct.keys())
    group_colors = {group: settings_dct[group][0] for group in groups_unique}
    group_labels = {group: settings_dct[group][1] for group in groups_unique}
    group_markers = {group: settings_dct[group][2] for group in groups_unique}

    reference_setting = "dfx_tsit5backsolve_1e-10_tol"
    if reference_setting not in df.index:
        print(f"Reference setting '{reference_setting}' not found in data.")
        return
    reference_loss = df.loc[reference_setting, "loss"]
    reference_grad = df.loc[reference_setting, "grad"]
    df = df.drop(reference_setting)

    df["loss_rmse"] = np.sqrt((df["loss"] - reference_loss) ** 2)
    df["loss_rmse"] = df["loss_rmse"].replace(0, np.nan)
    df["grad_rmse"] = np.sqrt((df["grad"] - reference_grad) ** 2)

    # group by group
    df_grouped = df.groupby("group")

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # iterate over groups and scatter
    for group in groups_unique:
        if group not in df_grouped.groups:
            continue
        df_group = df_grouped.get_group(group)
        c = [group_colors[group]] * len(df_group)
        m = group_markers[group]
        l = group_labels[group]

        axs[0, 0].scatter(
            df_group["runtime_fw"], df_group["loss_rmse"], c=c, marker=m, label=l
        )
        # plot with dashed line
        data = df_group[["runtime_fw", "loss_rmse"]]
        # sort data by runtime_fw
        data = data.sort_values(by="runtime_fw")
        axs[0, 0].plot(
            data["runtime_fw"],
            data["loss_rmse"],
            color=group_colors[group],
            linestyle="--",
            alpha=0.5,
        )
        # Annotate each point using the index as the label
        if DO_ANNOTATE:
            for i in range(len(df_group)):
                axs[0, 0].annotate(
                    df_group["spec"][i],
                    (df_group["runtime_fw"][i], df_group["loss_rmse"][i]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    ha="left",
                    fontsize=6,
                )

        axs[0, 1].scatter(
            df_group["runtime_bw"], df_group["grad_rmse"], c=c, marker=m
        )
        # plot with dashed line
        data = df_group[["runtime_bw", "grad_rmse"]]
        # sort data by runtime_bw
        data = data.sort_values(by="runtime_bw")
        axs[0, 1].plot(
            data["runtime_bw"],
            data["grad_rmse"],
            color=group_colors[group],
            linestyle="--",
            alpha=0.5,
        )
        # Annotate each point using the index as the label
        if DO_ANNOTATE:
            for i in range(len(df_group)):
                axs[0, 1].annotate(
                    df_group["spec"][i],
                    (df_group["runtime_bw"][i], df_group["grad_rmse"][i]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    ha="left",
                    fontsize=6,
                )

        axs[1, 0].scatter(
            df_group["jit_time_fw"], df_group["loss_rmse"], c=c, marker=m
        )

        axs[1, 1].scatter(
            df_group["jit_time_bw"], df_group["grad_rmse"], c=c, marker=m
        )

    axs[0, 0].set_xlabel("Runtime Forward (s)")
    axs[0, 0].set_ylabel("Loss RMSE")
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_xscale("log")

    axs[0, 1].set_xlabel("Runtime Backward (s)")
    axs[0, 1].set_ylabel("Grad RMSE")
    axs[0, 1].set_yscale("log")
    axs[0, 1].set_xscale("log")

    axs[1, 0].set_xlabel("JIT Time Forward (s)")
    axs[1, 0].set_ylabel("Loss RMSE")
    axs[1, 0].set_yscale("log")
    axs[1, 0].set_xscale("log")

    axs[1, 1].set_xlabel("JIT Time Backward (s)")
    axs[1, 1].set_ylabel("Grad RMSE")
    axs[1, 1].set_yscale("log")
    axs[1, 1].set_xscale("log")

    # Add a single legend outside all subplots
    fig.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),  # Position below the figure
        ncol=4,  # Spread legend entries across 2 columns
        frameon=True,
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])  # Leave space at the bottom
    # save figure
    plt.savefig(RESULTS_DIR / "bounce_scatter.pdf")
    plt.savefig(RESULTS_DIR / "bounce_scatter.png")


if __name__ == "__main__":
    main()
