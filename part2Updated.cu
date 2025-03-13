// Bradley Stephen | 19bbs2 | 20207842
// March 4th, 2025
// ELEC 374 - Machine Problem 1

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <time.h>

#define TOLERANCE 1e-3f
#define NUM_ITER 10     // Number of iterations per configuration (for Part 1 & 3)
#define SINGLETHREAD 5 // For single-thread experiments (Part 2)

/* ----------------- GPU Kernels ----------------- */

// Kernel for parallel execution (used in Part 3).
// Each thread computes one element of the output matrix.
__global__ void gpuMatrixMultiply(const float* matA, const float* matB, float* matC, int dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < dim && col < dim) {
        float sum = 0.0f;
        for (int k = 0; k < dim; ++k) {
            sum += matA[row * dim + k] * matB[k * dim + col];
        }
        matC[row * dim + col] = sum;
    }
}

// Kernel for serial execution (used in Part 2: Single-block, one-thread).
__global__ void gpuMatrixMultiplySingleThread(const float* matA, const float* matB, float* matC, int dim) {
    // A single thread computes the entire matrix.
    for (int row = 0; row < dim; ++row) {
        for (int col = 0; col < dim; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < dim; ++k) {
                sum += matA[row * dim + k] * matB[k * dim + col];
            }
            matC[row * dim + col] = sum;
        }
    }
}

/* ----------------- CPU Function ----------------- */

// CPU matrix multiplication.
void cpuMatrixMultiply(const float* matA, const float* matB, float* matC, int dim) {
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < dim; ++k) {
                sum += matA[i * dim + k] * matB[k * dim + j];
            }
            matC[i * dim + j] = sum;
        }
    }
}

// Compare two matrices element-wise within a tolerance.
int compareMatrices(const float* mat1, const float* mat2, int dim, float tol) {
    int totalElements = dim * dim;
    for (int i = 0; i < totalElements; ++i) {
        if (fabs(mat1[i] - mat2[i]) > tol)
            return 0;
    }
    return 1;
}

/* ----------------- Experiment Functions ----------------- */

// Part 1: Data Transfer Experiment: Measure H2D and D2H transfer times over NUM_ITER iterations.
void runDataTransferExperiment(int dim, FILE* csvTransfer) {
    size_t totalBytes = dim * dim * sizeof(float);
    float* hostA = (float*)malloc(totalBytes);
    float* hostB = (float*)malloc(totalBytes);
    if (!hostA || !hostB) {
        printf("Failed to allocate host memory for data transfer experiment for dimension %d.\n", dim);
        exit(EXIT_FAILURE);
    }
    srand((unsigned)time(NULL));
    for (int i = 0; i < dim * dim; ++i) {
        hostA[i] = (float)(rand() % 100) / 10.0f;
        hostB[i] = (float)(rand() % 100) / 10.0f;
    }
    float* devA, * devB;
    cudaMalloc((void**)&devA, totalBytes);
    cudaMalloc((void**)&devB, totalBytes);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    float sumH2D = 0, sumD2H = 0, sqSumH2D = 0, sqSumD2H = 0;

    for (int iter = 0; iter < NUM_ITER; ++iter) {
        // Measure Host-to-Device transfer time.
        cudaEventRecord(startEvent, 0);
        cudaMemcpy(devA, hostA, totalBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(devB, hostB, totalBytes, cudaMemcpyHostToDevice);
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        float h2dTime = 0.0f;
        cudaEventElapsedTime(&h2dTime, startEvent, stopEvent);

        // Measure Device-to-Host transfer time.
        float* hostResultA = (float*)malloc(totalBytes);
        float* hostResultB = (float*)malloc(totalBytes);
        cudaEventRecord(startEvent, 0);
        cudaMemcpy(hostResultA, devA, totalBytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(hostResultB, devB, totalBytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        float d2hTime = 0.0f;
        cudaEventElapsedTime(&d2hTime, startEvent, stopEvent);

        sumH2D += h2dTime;
        sumD2H += d2hTime;
        sqSumH2D += h2dTime * h2dTime;
        sqSumD2H += d2hTime * d2hTime;

        free(hostResultA);
        free(hostResultB);
    }

    float avgH2D = sumH2D / NUM_ITER;
    float avgD2H = sumD2H / NUM_ITER;
    float stdH2D = sqrt(sqSumH2D / NUM_ITER - avgH2D * avgH2D);
    float stdD2H = sqrt(sqSumD2H / NUM_ITER - avgD2H * avgD2H);

    printf("Data Transfer Experiment: Matrix %dx%d: H2D = %.3f ms, Std = %.3f ms, D2H = %.3f ms, Std = %.3f ms\n",
        dim, dim, avgH2D, stdH2D, avgD2H, stdD2H);
    fprintf(csvTransfer, "%d,%.3f,%.3f,%.3f,%.3f\n", dim, avgH2D, stdH2D, avgD2H, stdD2H);

    free(hostA);
    free(hostB);
    cudaFree(devA);
    cudaFree(devB);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

// Part 2: Single-Thread Experiment (GPU vs CPU)
// Uses a single block and one thread; CPU time is measured solely for comparison.
// This version runs the experiment for 'SINGLETHREAD' iterations and computes average and standard deviation.
void runSingleThreadExperiment(int dim, FILE* csvSingleThread) {
    size_t totalBytes = dim * dim * sizeof(float);
    float* hostA = (float*)malloc(totalBytes);
    float* hostB = (float*)malloc(totalBytes);
    float* hostC_gpu = (float*)malloc(totalBytes);
    float* hostC_cpu = (float*)malloc(totalBytes);
    if (!hostA || !hostB || !hostC_gpu || !hostC_cpu) {
        printf("Failed to allocate host memory for single-thread experiment for dimension %d.\n", dim);
        exit(EXIT_FAILURE);
    }
    srand((unsigned)time(NULL));
    for (int i = 0; i < dim * dim; ++i) {
        hostA[i] = (float)(rand() % 100) / 10.0f;
        hostB[i] = (float)(rand() % 100) / 10.0f;
    }
    float* devA, * devB, * devC;
    cudaMalloc((void**)&devA, totalBytes);
    cudaMalloc((void**)&devB, totalBytes);
    cudaMalloc((void**)&devC, totalBytes);
    cudaMemcpy(devA, hostA, totalBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, hostB, totalBytes, cudaMemcpyHostToDevice);

    float sumKernel = 0, sumCpu = 0;
    float sqSumKernel = 0, sqSumCpu = 0;
    int allTestsPassed = 1;
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    for (int iter = 0; iter < SINGLETHREAD; ++iter) {
        // Run the single-thread kernel.
        cudaEventRecord(startEvent, 0);
        gpuMatrixMultiplySingleThread << <1, 1 >> > (devA, devB, devC, dim);
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        float kernelTime = 0;
        cudaEventElapsedTime(&kernelTime, startEvent, stopEvent);
        // Copy result from device.
        cudaMemcpy(hostC_gpu, devC, totalBytes, cudaMemcpyDeviceToHost);
        // Measure CPU matrix multiplication time.
        clock_t cpuStart = clock();
        cpuMatrixMultiply(hostA, hostB, hostC_cpu, dim);
        clock_t cpuEnd = clock();
        float cpuTime = 1000.0f * (cpuEnd - cpuStart) / CLOCKS_PER_SEC;
        // Validate the results.
        int testPassed = compareMatrices(hostC_gpu, hostC_cpu, dim, TOLERANCE);
        if (!testPassed)
            allTestsPassed = 0;
        sumKernel += kernelTime;
        sumCpu += cpuTime;
        sqSumKernel += kernelTime * kernelTime;
        sqSumCpu += cpuTime * cpuTime;
    }

    float avgKernel = sumKernel / SINGLETHREAD;
    float avgCpu = sumCpu / SINGLETHREAD;
    float stdKernel = sqrt(sqSumKernel / SINGLETHREAD - avgKernel * avgKernel);
    float stdCpu = sqrt(sqSumCpu / SINGLETHREAD - avgCpu * avgCpu);

    printf("----- Matrix Dimension: %dx%d -----\n", dim, dim);
    printf("GPU Kernel Execution (Single Thread): Avg = %.3f ms, Std = %.3f ms\n", avgKernel, stdKernel);
    printf("CPU Computation Time: Avg = %.3f ms, Std = %.3f ms\n", avgCpu, stdCpu);
    if (allTestsPassed)
        printf("Test PASSED: GPU and CPU results are equivalent!\n\n");
    else
        printf("Test FAILED: GPU and CPU results do not match.\n\n");
    fprintf(csvSingleThread, "%d,%.3f,%.3f,%.3f,%.3f\n", dim, avgKernel, stdKernel, avgCpu, stdCpu);

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    free(hostA);
    free(hostB);
    free(hostC_gpu);
    free(hostC_cpu);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

// Part 3: Kernel-Only Experiment (Varying Block Sizes)
// Measures only the GPU kernel execution time (ignoring data transfers and memory allocation/free time).
void runKernelOnlyExperiment(int dim, int blockSize, FILE* csvKernel) {
    size_t totalBytes = dim * dim * sizeof(float);
    float* hostA = (float*)malloc(totalBytes);
    float* hostB = (float*)malloc(totalBytes);
    float* hostC_gpu = (float*)malloc(totalBytes);
    if (!hostA || !hostB || !hostC_gpu) {
        printf("Failed to allocate host memory for kernel-only experiment for dimension %d.\n", dim);
        exit(EXIT_FAILURE);
    }
    srand((unsigned)time(NULL));
    for (int i = 0; i < dim * dim; ++i) {
        hostA[i] = (float)(rand() % 100) / 10.0f;
        hostB[i] = (float)(rand() % 100) / 10.0f;
    }
    float* devA, * devB, * devC;
    cudaMalloc((void**)&devA, totalBytes);
    cudaMalloc((void**)&devB, totalBytes);
    cudaMalloc((void**)&devC, totalBytes);
    // Copy input matrices to device.
    cudaMemcpy(devA, hostA, totalBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, hostB, totalBytes, cudaMemcpyHostToDevice);

    // Set grid and block dimensions.
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 numBlocks((dim + blockSize - 1) / blockSize, (dim + blockSize - 1) / blockSize);

    // Timing loop: measure only the kernel execution time.
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float sum_kernel = 0, sq_sum_kernel = 0;
    for (int i = 0; i < NUM_ITER; ++i) {
        cudaEventRecord(startEvent, 0);
        gpuMatrixMultiply << <numBlocks, threadsPerBlock >> > (devA, devB, devC, dim);
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        float t = 0;
        cudaEventElapsedTime(&t, startEvent, stopEvent);
        sum_kernel += t;
        sq_sum_kernel += t * t;
    }
    float avg_kernel = sum_kernel / NUM_ITER;
    float std_kernel = sqrt(sq_sum_kernel / NUM_ITER - avg_kernel * avg_kernel);

    printf("----- Kernel-Only: Matrix %dx%d, Block Width: %d -----\n", dim, dim, blockSize);
    printf("GPU Kernel Execution: Avg = %.3f ms, Std = %.3f ms\n\n", avg_kernel, std_kernel);
    fprintf(csvKernel, "%d,%d,%.3f,%.3f\n", dim, blockSize, avg_kernel, std_kernel);

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    free(hostA);
    free(hostB);
    free(hostC_gpu);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

/* ----------------- Main ----------------- */

int main(void) {
    // Define matrix sizes and block sizes.
    int matrixSizes[5] = { 256, 512, 1024, 2048, 4096 };
    int matrixSizesST[3] = { 256, 512, 1024 }; // For Part 2 (single-thread)
    int blockSizes[5] = { 2, 4, 8, 16, 32 };

    // Data Transfer Experiment.
    /*FILE* csvTransfer = fopen("data_transfer_times.csv", "w");
    if (!csvTransfer) {
        printf("Error opening CSV file for data transfer experiments.\n");
        return EXIT_FAILURE;
    }
    // CSV header with units.
    fprintf(csvTransfer, "Matrix_Size,H2D_time (ms),H2D_std (ms),D2H_time (ms),D2H_std (ms)\n");
    for (int i = 0; i < 5; ++i) {
        runDataTransferExperiment(matrixSizes[i], csvTransfer);
    }
    fclose(csvTransfer);*/

    // Part 2: GPU vs CPU Comparison using Single-Thread Kernel.
    FILE* csvSingleThread = fopen("gpu_single_thread_times.csv", "w");
    if (!csvSingleThread) {
        printf("Error opening CSV file for single-thread experiments.\n");
        return EXIT_FAILURE;
    }
    // CSV header with units.
    fprintf(csvSingleThread, "Matrix_Size,GPU_SingleThread_time (ms),GPU_SingleThread_std (ms),CPU_time (ms),CPU_std (ms)\n");
    for (int i = 0; i < 3; ++i) {
        runSingleThreadExperiment(matrixSizesST[i], csvSingleThread);
    }
    fclose(csvSingleThread);

    // Part 3: Kernel-Only Experiments (Varying Block Sizes).
    /*FILE* csvKernel = fopen("kernel_times.csv", "w");
    if (!csvKernel) {
        printf("Error opening CSV file for kernel-only experiments.\n");
        return EXIT_FAILURE;
    }
    // CSV header with units.
    fprintf(csvKernel, "Matrix_Size,Block_Size,Kernel_time (ms),Kernel_std (ms)\n");
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            runKernelOnlyExperiment(matrixSizes[i], blockSizes[j], csvKernel);
        }
    }
    fclose(csvKernel);*/

    return 0;
}
