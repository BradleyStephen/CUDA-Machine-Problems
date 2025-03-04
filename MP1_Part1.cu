// Bradley Stephen | 19bbs2 | 20207842
// March 4th, 2025
// ELEC 374 - Machine Problem 1 - Part 1

#include <cuda_runtime.h>
#include <stdio.h>

// Query and prinrt CUDA device porps
void queryDeviceProperties() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("CUDA Devices Found: %d\n", deviceCount);

    if (deviceCount == 0) {
        printf("No CUDA devices available.\n");
        return;
    }

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        // Basic device info
        printf("\nDevice %d: %s\n", i, deviceProp.name);
        printf("Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("Clock Speed: %d kHz\n", deviceProp.clockRate);
        printf("Streaming Multiprocessors (SMs): %d\n", deviceProp.multiProcessorCount);

        // Determine CUDA core count based on specifc arc.
        int coresPerSM;
        switch (deviceProp.major) {
        case 2:  coresPerSM = (deviceProp.minor == 1) ? 48 : 32; break;  // Fermi
        case 3:  coresPerSM = 192; break;  // Kepler
        case 5:  coresPerSM = 128; break;  // Maxwell
        case 6:  coresPerSM = (deviceProp.minor == 1) ? 128 : 64; break;  // Pascal
        case 7:  coresPerSM = 64; break;  // Volta/Turing
        case 8:  coresPerSM = (deviceProp.minor == 6 || deviceProp.minor == 9) ? 128 : 64; break;  // Ampere
        default: coresPerSM = 64;  // Default: 64
        }

        // Print Results
        int totalCores = coresPerSM * deviceProp.multiProcessorCount;
        printf("CUDA Cores per SM: %d\n", coresPerSM);
        printf("Total CUDA Cores: %d\n", totalCores);
        printf("Warp Size: %d\n", deviceProp.warpSize);

        // Mem sizes
        printf("Global Memory: %.2f GB\n", deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("Constant Memory: %.2f KB\n", deviceProp.totalConstMem / 1024.0);
        printf("Shared Memory per Block: %.2f KB\n", deviceProp.sharedMemPerBlock / 1024.0);
        printf("Registers per Block: %d\n", deviceProp.regsPerBlock);
        printf("Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);

        // Max limits for parallel EX
        printf("Max Block Dimensions: (%d, %d, %d)\n",
            deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("Max Grid Dimensions: (%d, %d, %d)\n",
            deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    }
}

// Run query 
int main() {
    queryDeviceProperties();
    return 0;
}
