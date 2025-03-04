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


# Plot 2: Kernel Execution Time vs. Block Size 
df_kernel = pd.read_csv("kernel_times.csv")
matrix_sizes_unique = sorted(df_kernel["Matrix_Size"].unique())
fig, ax = plt.subplots(figsize=(8,6))
for size in matrix_sizes_unique:
    subset = df_kernel[df_kernel["Matrix_Size"] == size]
    subset = subset.sort_values("Block_Size")
    ax.plot(subset["Block_Size"], subset["Kernel_Time"], marker='o', label=f'{size}x{size}')

ax.set_xlabel("Block Size")
ax.set_ylabel("Kernel Execution Time (ms)")
ax.set_title("GPU Kernel Execution Time vs. Block Size")
ax.legend(title="Matrix Size")
plt.savefig("kernel_times.png", dpi=300)
plt.close()


# Plot 3: CPU vs. GPU Computation Time vs. Matrix Size 
df_cpu_gpu = pd.read_csv("cpu_vs_gpu_times.csv")
df_cpu_gpu_avg = df_cpu_gpu.groupby("Matrix_Size", as_index=False).mean()
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(df_cpu_gpu_avg["Matrix_Size"], df_cpu_gpu_avg["CPU_Time"], marker='o', label="CPU Time")
ax.plot(df_cpu_gpu_avg["Matrix_Size"], df_cpu_gpu_avg["GPU_Time"], marker='o', label="GPU Kernel Time")
ax.set_xlabel("Matrix Size (n x n)")
ax.set_ylabel("Computation Time (ms)")
ax.set_title("CPU vs. GPU Computation Time")
ax.legend()
plt.savefig("cpu_vs_gpu_times.png", dpi=300)
plt.close()
