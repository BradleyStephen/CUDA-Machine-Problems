import pandas as pd
import matplotlib.pyplot as plt

# Read the bonus CSV file
df_bonus = pd.read_csv("MP2Bonus.csv")
print("Bonus CSV data loaded successfully. Here is a preview:")
print(df_bonus.head())

# Expected CSV columns for bonus:
# Test, A_rows, A_cols, B_cols, CPUTimeMs, GPUKernelAvgMs, GPUKernelStdMs, Result

# Plot 1: Bar Plot Comparing CPU vs GPU Kernel Time for each Test Case
plt.figure(figsize=(10, 6))
# Convert test case number to string for x-axis labeling
tests = df_bonus["Test"].astype(str)
cpu_time = df_bonus["CPUTimeMs"]
gpu_time = df_bonus["GPUKernelAvgMs"]
gpu_std = df_bonus["GPUKernelStdMs"]

# Define bar width and positions
bar_width = 0.35
indices = range(len(tests))

# Plot CPU times and GPU kernel times (with error bars)
plt.bar(indices, cpu_time, bar_width, label="CPU Reference Time", color="red")
plt.bar([i + bar_width for i in indices], gpu_time, bar_width, yerr=gpu_std, 
        capsize=5, label="GPU Kernel Time", color="blue")

plt.xlabel("Test Case")
plt.ylabel("Time (ms)")
plt.title("CPU vs GPU Kernel Time for Bonus Test Cases")
plt.xticks([i + bar_width/2 for i in indices], tests)
plt.legend()
plt.tight_layout()
plt.savefig("bonus_cpu_vs_gpu.png")
plt.show()

# Plot 2: Scatter Plot of GPU Kernel Time vs. Matrix M Rows (A_rows) with error bars
plt.figure(figsize=(10, 6))
plt.errorbar(df_bonus["A_rows"], df_bonus["GPUKernelAvgMs"],
             yerr=df_bonus["GPUKernelStdMs"], fmt="o", capsize=5)
plt.xlabel("Matrix M Rows (A_rows)")
plt.ylabel("GPU Kernel Time (ms)")
plt.title("GPU Kernel Time vs. Matrix M Rows (Bonus Test Cases)")
plt.grid(True)
plt.tight_layout()
plt.savefig("bonus_gpu_kernel_vs_rows.png")
plt.show()
