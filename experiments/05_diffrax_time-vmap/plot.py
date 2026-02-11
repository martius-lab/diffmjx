import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parent directory that contains the batch folders
DIRPATH = os.path.dirname(os.path.abspath(__file__))


NUM_SPHERES = 10
#NUM_BATCH = 2
#batch_nums = list(range(1,NUM_BATCH+1))   # 0..9
batch_nums = [1, 10]#, 100, 1000]
data = []

for b in batch_nums:
    for used_diffrax in [False]:
        folder_name = f"bw-diffrax={used_diffrax}-cfd=False-num_spheres={NUM_SPHERES}-num_batch={b}"
        folder_path = os.path.join(DIRPATH, "outputs", "plots", folder_name)

        # Find the CSV file in this folder (assumes exactly one CSV per folder)
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        if not csv_files:
            print(f"No CSV found in {folder_path}, skipping.")
            continue

        csv_path = csv_files[0]

        # First column in your example is just an index, so use index_col=0
        df = pd.read_csv(csv_path, index_col=0)

        # Take the first row (assuming one row per file)
        runtime = df["runtime"].iloc[0]
        grad = df["grad"].iloc[0]
        grad_fd = df["grad_fd"].iloc[0]
        grad_err = np.abs(df["grad_fd"].iloc[0] - df["grad"].iloc[0])
        runs_per_second = df["runs_per_second"].iloc[0]
        total_runs_per_second = df["total_runs_per_second"].iloc[0]

        data.append((b, runtime, grad, grad_fd, grad_err, runs_per_second, total_runs_per_second))

# Put into a DataFrame and sort by batch number
if not data:
    raise RuntimeError("No data collected; check paths / filenames.")

result_df = pd.DataFrame(data, columns=["batch_num", "runtime", "grad_err", "grad", "grad_fd", "runs_per_second", "total_runs_per_second"])
result_df = result_df.sort_values("batch_num")

fig, ax = plt.subplots(figsize=(7, 4), dpi=120)

ax.plot(
    result_df["batch_num"],
    result_df["runs_per_second"],
    marker="o",
    linewidth=2,
    linestyle='None',
    markersize=8,
    color="#1f77b4",
)
ax.plot(
    result_df["batch_num"],
    result_df["grad_err"],
    marker="o",
    linewidth=2,
    linestyle='None',
    markersize=8,
    color="#1f77b4",
)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Number of simulations run in parallel", fontsize=11)
ax.set_ylabel("Runtime in seconds", fontsize=11)

# Remove top and right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Make x-ticks exactly the batch numbers
ax.set_xticks(sorted(result_df["batch_num"].unique()))
ax.set_xticks([1,10,100,1000])
#ax.set_yticks([0,10,20])
plt.tight_layout()
plot_path = os.path.join(DIRPATH, "outputs", "plots", "plot.pdf")
plt.savefig(plot_path)
#plt.show()