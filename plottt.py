import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file produced by the CUDA program
df = pd.read_csv("MP2.csv")
print("CSV data loaded successfully. Here is a preview:")
print(df.head())

# Plot 1: GPU Kernel Execution Time vs. Matrix Size (for different tile widths)
plt.figure(figsize=(10, 6))
tile_widths = sorted(df["TileWidth"].unique())
for tw in tile_widths:
    subset = df[df["TileWidth"] == tw].sort_values("MatrixSize")
    plt.plot(subset["MatrixSize"], subset["GPUKernelTimeMs"], marker="o", label=f"TileWidth {tw}")

plt.xlabel("Matrix Size (n x n)")
plt.ylabel("GPU Kernel Time (ms)")
plt.title("GPU Kernel Time vs. Matrix Size (by Tile Width)")
plt.legend(title="Tile Width")
plt.grid(True)
plt.tight_layout()
plt.savefig("gpu_kernel_time.png")
plt.show()

# Plot 2: CPU Reference Time vs. Matrix Size
# CPUTimeMs is the same for a given matrix size, so we take unique values per size.
cpu_df = df.drop_duplicates(subset=["MatrixSize"]).sort_values("MatrixSize")
plt.figure(figsize=(10, 6))
plt.plot(cpu_df["MatrixSize"], cpu_df["CPUTimeMs"], marker="o", color="red", label="CPU Reference Time")
plt.xlabel("Matrix Size (n x n)")
plt.ylabel("CPU Time (ms)")
plt.title("CPU Reference Time vs. Matrix Size")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cpu_reference_time.png")
plt.show()

# Optionally, Plot 3: Host-to-Device Transfer Time vs. Matrix Size for each tile width
plt.figure(figsize=(10, 6))
for tw in tile_widths:
    subset = df[df["TileWidth"] == tw].sort_values("MatrixSize")
    plt.plot(subset["MatrixSize"], subset["HostToDeviceMs"], marker="o", linestyle="--", label=f"H->D TileWidth {tw}")

plt.xlabel("Matrix Size (n x n)")
plt.ylabel("Host-to-Device Transfer Time (ms)")
plt.title("Host-to-Device Transfer Time vs. Matrix Size")
plt.legend(title="Tile Width")
plt.grid(True)
plt.tight_layout()
plt.savefig("host_to_device_time.png")
plt.show()

# Optionally, Plot 4: Device-to-Host Transfer Time vs. Matrix Size for each tile width
plt.figure(figsize=(10, 6))
for tw in tile_widths:
    subset = df[df["TileWidth"] == tw].sort_values("MatrixSize")
    plt.plot(subset["MatrixSize"], subset["DeviceToHostMs"], marker="o", linestyle="--", label=f"D->H TileWidth {tw}")

plt.xlabel("Matrix Size (n x n)")
plt.ylabel("Device-to-Host Transfer Time (ms)")
plt.title("Device-to-Host Transfer Time vs. Matrix Size")
plt.legend(title="Tile Width")
plt.grid(True)
plt.tight_layout()
plt.savefig("device_to_host_time.png")
plt.show()
