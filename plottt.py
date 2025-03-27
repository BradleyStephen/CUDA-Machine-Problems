import pandas as pd
import matplotlib.pyplot as plt

# Read bonus CSV file (MP2Bonus.csv)
df_bonus = pd.read_csv("MP2Bonus.csv")
print("Bonus CSV loaded successfully. Preview:")
print(df_bonus.head())

# Expected columns in MP2Bonus.csv:
# Test, A_rows, A_cols, B_cols, CPUTimeMs, GPUKernelAvgMs, GPUKernelStdMs, Result

# Plot 1: Bar Plot Comparing CPU and GPU Kernel Times for Each Test Case
plt.figure(figsize=(8,6))
tests = df_bonus["Test"].astype(str)
cpu_times = df_bonus["CPUTimeMs"]
gpu_times = df_bonus["GPUKernelAvgMs"]
gpu_std = df_bonus["GPUKernelStdMs"]

bar_width = 0.35
indices = range(len(tests))

plt.bar(indices, cpu_times, bar_width, color="salmon", label="CPU Time (ms)")
plt.bar([i + bar_width for i in indices], gpu_times, bar_width, yerr=gpu_std, capsize=5, 
        color="skyblue", label="GPU Kernel Time (ms)")

plt.xlabel("Test Case")
plt.ylabel("Time (ms)")
plt.title("Bonus: CPU vs GPU Kernel Times for Each Test Case")
plt.xticks([i + bar_width/2 for i in indices], tests)
plt.legend()
plt.tight_layout()
plt.savefig("bonus_cpu_vs_gpu.png")
plt.show()

# Plot 2: Scatter Plot of GPU Kernel Time vs. Matrix M Rows (A_rows)
plt.figure(figsize=(8,6))
plt.errorbar(df_bonus["A_rows"], df_bonus["GPUKernelAvgMs"], 
             yerr=df_bonus["GPUKernelStdMs"], fmt="o", capsize=5, color="blue")
plt.xlabel("Matrix M Rows (A_rows)")
plt.ylabel("GPU Kernel Time (ms)")
plt.title("Bonus: GPU Kernel Time vs. Matrix M Rows")
plt.grid(True)
plt.tight_layout()
plt.savefig("bonus_gpu_kernel_vs_rows.png")
plt.show()
