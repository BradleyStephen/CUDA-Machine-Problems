// Bradley Stephen | 19bbs2 | 20207842
// March 3rd, 2025
// Machine Problem 1 - Part 1 - ELEC 374
#include <cuda_runtime.h>
#include <stdio.h>

void queryDeviceProperties() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Number of CUDA devices: %d\n", deviceCount);

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return;
    }

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("\nDevice %d: %s\n", i, prop.name);
        printf("CUDA Capability: %d.%d\n", prop.major, prop.minor);
        printf("Clock Rate: %d kHz\n", prop.clockRate);
        printf("Number of Streaming Multiprocessors (SMs): %d\n", prop.multiProcessorCount);

        // Compute number of CUDA cores (Using computer in bain lab)
        int coresPerSM;
        switch (prop.major) {
        case 2: // Fermi
            coresPerSM = (prop.minor == 1) ? 48 : 32;
            break;
        case 3: // Kepler
            coresPerSM = 192;
            break;
        case 5: // Maxwell
            coresPerSM = 128;
            break;
        case 6: // Pascal
            coresPerSM = (prop.minor == 1) ? 128 : 64;
            break;
        case 7: // Volta/Turing
            coresPerSM = 64;
            break;
        case 8: // Ampere
            coresPerSM = (prop.minor == 6 || prop.minor == 9) ? 128 : 64;
            break;
        default: // Unknown architecture
            coresPerSM = 64;
        }
        printf("CUDA Cores per SM: %d\n", coresPerSM);
        printf("Total CUDA Cores: %d\n", coresPerSM * prop.multiProcessorCount);

        printf("Warp Size: %d\n", prop.warpSize);
        printf("Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("Constant Memory: %.2f KB\n", prop.totalConstMem / 1024.0);
        printf("Shared Memory per Block: %.2f KB\n", prop.sharedMemPerBlock / 1024.0);
        printf("Registers per Block: %d\n", prop.regsPerBlock);
        printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("Max Block Dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max Grid Dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("----------------------------------------------------\n");
    }
}

int main() {
    queryDeviceProperties();
    return 0;
}
