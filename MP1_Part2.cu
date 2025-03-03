// Bradley Stephen | 19bbs2 | 20207842
// Machine problem 1 | Part 1 | ELEC 374
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "device_launch_parameters.h"

// GPU Kernel for Matrix Multiplication
// Each thread computes one element of the output matrix.
__global__ void gpuMatrixMultiplyKernel(float* d_P, float* d_M, float* d_N, int matrixDim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < matrixDim && col < matrixDim) {
        float value = 0.0f;
        for (int k = 0; k < matrixDim; k++) {
            value += d_M[row * matrixDim + k] * d_N[k * matrixDim + col];
        }
        d_P[row * matrixDim + col] = value;
    }
}

// CPU Implementation for Matrix Multiplication (for verification)
void cpuMatrixMultiply(float* result, float* A, float* B, int matrixDim) {
    for (int i = 0; i < matrixDim; i++) {
        for (int j = 0; j < matrixDim; j++) {
            float sum = 0.0f;
            for (int k = 0; k < matrixDim; k++) {
                sum += A[i * matrixDim + k] * B[k * matrixDim + j];
            }
            result[i * matrixDim + j] = sum;
        }
    }
}

// Compare two matrices with a given tolerance.
bool verifyMatrices(float* mat1, float* mat2, int matrixDim, float tol) {
    for (int i = 0; i < matrixDim * matrixDim; i++) {
        if (fabsf(mat1[i] - mat2[i]) > tol) {
            return false;
        }
    }
    return true;
}


/* Main runs experiments on different matrix and block sizes, measures data
   transfer times, kernel execution, and CPU multiplication for verification.*/
int main() {
    // Experiment with various matrix dimensions and CUDA block widths for analysis.
    int matrixSizes[] = { 256, 512, 1024, 2048, 4096 };
    int blockWidths[] = { 2, 4, 8, 16, 32 };

    // Files for recording timing data and to use in plots.
    FILE* fileH2D = fopen("data_transfers_logs.txt", "w");
    FILE* fileKernel = fopen("kernel_time_logs.txt", "w");
    FILE* fileCPUvsGPU = fopen("cpu_gpu_time_logs.txt", "w");

    fprintf(fileH2D, "Matrix_Size,Host_to_Device,Device_to_Host\n");
    fprintf(fileKernel, "Matrix_Size,Block_Size,GPU_Kernel_Time\n");
    fprintf(fileCPUvsGPU, "Matrix_Size,CPU_Time,GPU_Time\n");

    // Loop over different matrix sizes.
    for (int ms = 0; ms < 5; ms++) {
        // Loop over different block sizes.
        for (int bs = 0; bs < 5; bs++) {
            int matrixDim = matrixSizes[ms];
            int blockSize = blockWidths[bs];
            size_t bytes = matrixDim * matrixDim * sizeof(float);

            printf("\n===== Matrix Dimension: %d x %d, Block Width: %d =====\n", matrixDim, matrixDim, blockSize);

            // Allocate host memory.
            float* h_A = (float*)malloc(bytes);
            float* h_B = (float*)malloc(bytes);
            float* h_gpuResult = (float*)malloc(bytes);
            float* h_cpuResult = (float*)malloc(bytes);
            if (!h_A || !h_B || !h_gpuResult || !h_cpuResult) {
                printf("Error allocating host memory.\n");
                return -1;
            }

            // Initialize input matrices with random float values.
            srand((unsigned)time(NULL));
            for (int i = 0; i < matrixDim * matrixDim; i++) {
                h_A[i] = (float)(rand() % 100) / 10.0f;
                h_B[i] = (float)(rand() % 100) / 10.0f;
            }

            // Allocate device memory.
            float* d_A, * d_B, * d_P;
            cudaMalloc((void**)&d_A, bytes);
            cudaMalloc((void**)&d_B, bytes);
            cudaMalloc((void**)&d_P, bytes);

            // Create CUDA events for timing.
            cudaEvent_t startEvent, stopEvent;
            cudaEventCreate(&startEvent);
            cudaEventCreate(&stopEvent);
            float elapsedTime = 0.0f;

            // Measure Host-to-Device memory copy time.
            cudaEventRecord(startEvent, 0);
            cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
            cudaEventRecord(stopEvent, 0);
            cudaEventSynchronize(stopEvent);
            cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
            printf("Host-to-Device Transfer: %.3f ms\n", elapsedTime);
            float h2dTime = elapsedTime;

            // Setup execution configuration.
            dim3 threads(blockSize, blockSize);
            dim3 blocks((matrixDim + blockSize - 1) / blockSize,
                (matrixDim + blockSize - 1) / blockSize);

            // Execute the GPU kernel and time it.
            cudaEventRecord(startEvent, 0);
            gpuMatrixMultiplyKernel<<<blocks, threads >>>(d_P, d_A, d_B, matrixDim);
            cudaEventRecord(stopEvent, 0);
            cudaEventSynchronize(stopEvent);
            cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
            printf("GPU Kernel Execution: %.3f ms\n", elapsedTime);
            float kernelTime = elapsedTime;

            // Measure Device-to-Host memory copy time.
            cudaEventRecord(startEvent, 0);
            cudaMemcpy(h_gpuResult, d_P, bytes, cudaMemcpyDeviceToHost);
            cudaEventRecord(stopEvent, 0);
            cudaEventSynchronize(stopEvent);
            cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
            printf("Device-to-Host Transfer: %.3f ms\n", elapsedTime);
            float d2hTime = elapsedTime;

            // CPU-based matrix multiplication for validation.
            clock_t cpuStart = clock();
            cpuMatrixMultiply(h_cpuResult, h_A, h_B, matrixDim);
            clock_t cpuEnd = clock();
            float cpuTime = 1000.0f * (cpuEnd - cpuStart) / CLOCKS_PER_SEC;
            printf("CPU Computation Time: %.3f ms\n", cpuTime);

            // Validate the GPU result against the CPU result.
            if (verifyMatrices(h_gpuResult, h_cpuResult, matrixDim, 1e-3f)) {
                printf("Test PASSED: GPU and CPU results are equivalent!\n");
            }
            else {
                printf("Test FAILED: Mismatch between GPU and CPU results.\n");
            }

            // Write timing results to files.
            fprintf(fileH2D, "%d,%.3f,%.3f\n", matrixDim, h2dTime, d2hTime);
            fprintf(fileKernel, "%d,%d,%.3f\n", matrixDim, blockSize, kernelTime);
            fprintf(fileCPUvsGPU, "%d,%.3f,%.3f\n", matrixDim, cpuTime, kernelTime);

            // Free allocated resources.
            free(h_A);
            free(h_B);
            free(h_gpuResult);
            free(h_cpuResult);
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_P);
            cudaEventDestroy(startEvent);
            cudaEventDestroy(stopEvent);
        }
    }

    fclose(fileH2D);
    fclose(fileKernel);
    fclose(fileCPUvsGPU);

    return 0;
}
