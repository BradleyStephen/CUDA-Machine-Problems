import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame.
# Expected CSV columns: "Matrix_Size", "GPU_SingleThread_time (ms)", "GPU_SingleThread_std (ms)",
# "CPU_time (ms)", "CPU_std (ms)"
df = pd.read_csv("gpu_single_thread_times.csv")

# Create a figure with two subplots side by side.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

# Plot GPU single-thread times.
ax1.errorbar(df["Matrix_Size"], df["GPU_SingleThread_time (ms)"],
             yerr=df["GPU_SingleThread_std (ms)"],
             fmt='-o', capsize=5, color='blue')
ax1.set_xlabel("Matrix Size")
ax1.set_ylabel("Time (ms)")
ax1.set_title("GPU Single-Thread Times")
ax1.grid(True)

# Plot CPU times.
ax2.errorbar(df["Matrix_Size"], df["CPU_time (ms)"],
             yerr=df["CPU_std (ms)"],
             fmt='-s', capsize=5, color='red')
ax2.set_xlabel("Matrix Size")
ax2.set_title("CPU Times")
ax2.grid(True)

plt.suptitle("Single-Thread Matrix Multiplication: GPU vs CPU", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("gpu_cpu_single_thread_times_side_by_side.png", dpi=300)
plt.show()
