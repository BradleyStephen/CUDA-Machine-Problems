# Im so bad at excel lol using python to plot neatly
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Plot 1: Data Transfer Times
df_transfer = pd.read_csv("data_transfer_times.csv")
df_transfer_avg = df_transfer.groupby("Matrix_Size", as_index=False).mean()
matrix_sizes = df_transfer_avg["Matrix_Size"].astype(str)
x = np.arange(len(matrix_sizes))  # label locations
width = 0.35  
fig, ax = plt.subplots(figsize=(8,6))
bar1 = ax.bar(x - width/2, df_transfer_avg["HostToDevice"], width, label="Host-to-Device")
bar2 = ax.bar(x + width/2, df_transfer_avg["DeviceToHost"], width, label="Device-to-Host")
ax.set_xlabel("Matrix Size (n x n)")
ax.set_ylabel("Transfer Time (ms)")
ax.set_title("Data Transfer Times vs. Matrix Size")
ax.set_xticks(x)
ax.set_xticklabels(matrix_sizes)
ax.legend()
plt.savefig("data_transfer_times.png", dpi=300)
plt.close()


