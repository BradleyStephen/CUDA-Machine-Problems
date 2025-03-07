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
#define NUM_ITER 10  // Number of iterations per configuration
#define SINGLETHREAD 5

// ----------------- GPU Kernels -----------------

// GPU kernel for parallel execution (used in Part 3).
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

// GPU kernel for serial execution (used in Part 2: Single block, one thread).
__global__ void gpuMatrixMultiplySingleThread(const float* matA, const float* matB, float* matC, int dim) {
    // Single thread computes the entire matrix.
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

// ----------------- CPU Function -----------------

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

// Compare matrices with a tolerance.
int compareMatrices(const float* mat1, const float* mat2, int dim, float tol) {
    int totalElements = dim * dim;
    for (int i = 0; i < totalElements; ++i) {
        if (fabs(mat1[i] - mat2[i]) > tol)
            return 0;
    }
    return 1;
}

// ----------------- Experiment Functions -----------------

// Data Transfer Experiment: Measure H2D and D2H transfer times.
void runDataTransferExperiment(int dim, FILE* csvTransfer) {
    size_t totalBytes = dim * dim * sizeof(float);

    // Allocate and initialize host matrices.
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

    // Allocate device memory.
    float* devA, * devB;
    cudaMalloc((void**)&devA, totalBytes);
    cudaMalloc((void**)&devB, totalBytes);

    // Create CUDA events.
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // Measure Host-to-Device transfer time.
    cudaEventRecord(startEvent, 0);
    cudaMemcpy(devA, hostA, totalBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, hostB, totalBytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float h2dTime = 0.0f;
    cudaEventElapsedTime(&h2dTime, startEvent, stopEvent);

    // Prepare host buffers for D2H.
    float* hostResultA = (float*)malloc(totalBytes);
    float* hostResultB = (float*)malloc(totalBytes);

    // Measure Device-to-Host transfer time.
    cudaEventRecord(startEvent, 0);
    cudaMemcpy(hostResultA, devA, totalBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostResultB, devB, totalBytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float d2hTime = 0.0f;
    cudaEventElapsedTime(&d2hTime, startEvent, stopEvent);

    printf("Data Transfer Experiment: Matrix %dx%d: H2D = %.3f ms, D2H = %.3f ms\n", dim, dim, h2dTime, d2hTime);
    fprintf(csvTransfer, "%d,%.3f,%.3f\n", dim, h2dTime, d2hTime);

    // Clean up.
    free(hostA);
    free(hostB);
    free(hostResultA);
    free(hostResultB);
    cudaFree(devA);
    cudaFree(devB);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

// Part 2: Single-Thread Experiment (GPU vs CPU)
// GPU computation is performed using a single block and one thread.
// Note: Kernel metrics printing is removed from this part.
void runSingleThreadExperiment(int dim, FILE* csvSingleThread) {
    size_t totalBytes = dim * dim * sizeof(float);

    // Allocate and initialize host matrices.
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

    // Allocate device memory.
    float* devA, * devB, * devC;
    cudaMalloc((void**)&devA, totalBytes);
    cudaMalloc((void**)&devB, totalBytes);
    cudaMalloc((void**)&devC, totalBytes);

    // Copy matrices to device.
    cudaMemcpy(devA, hostA, totalBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, hostB, totalBytes, cudaMemcpyHostToDevice);

    // (No kernel metrics printing here for Part 2.)

    float sum_kernel = 0, sum_cpu = 0;
    float sq_sum_kernel = 0, sq_sum_cpu = 0;

    // Create CUDA events for timing.
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    for (int iter = 0; iter < SINGLETHREAD; ++iter) {
        cudaEventRecord(startEvent, 0);
        gpuMatrixMultiplySingleThread << <1, 1 >> > (devA, devB, devC, dim);
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        float kernelTime = 0.0f;
        cudaEventElapsedTime(&kernelTime, startEvent, stopEvent);

        // Copy result back to host.
        cudaMemcpy(hostC_gpu, devC, totalBytes, cudaMemcpyDeviceToHost);

        // Measure CPU matrix multiplication time.
        clock_t cpuStart = clock();
        cpuMatrixMultiply(hostA, hostB, hostC_cpu, dim);
        clock_t cpuEnd = clock();
        float cpuTime = 1000.0f * (cpuEnd - cpuStart) / CLOCKS_PER_SEC;

        // Validate results.
        if (!compareMatrices(hostC_gpu, hostC_cpu, dim, TOLERANCE)) {
            printf("Test FAILED: GPU and CPU results do not match for dimension %d on iteration %d.\n", dim, iter);
        }

        sum_kernel += kernelTime;
        sum_cpu += cpuTime;
        sq_sum_kernel += kernelTime * kernelTime;
        sq_sum_cpu += cpuTime * cpuTime;
    }

    float avg_kernel = sum_kernel / SINGLETHREAD;
    float avg_cpu = sum_cpu / SINGLETHREAD;
    float std_kernel = sqrt(sq_sum_kernel / SINGLETHREAD - avg_kernel * avg_kernel);
    float std_cpu = sqrt(sq_sum_cpu / SINGLETHREAD - avg_cpu * avg_cpu);

    printf("----- Single-Thread Config: Matrix %dx%d -----\n", dim, dim);
    printf("GPU Kernel Execution (Single Thread): Avg = %.3f ms, StdDev = %.3f ms\n", avg_kernel, std_kernel);
    printf("CPU Computation Time              : Avg = %.3f ms, StdDev = %.3f ms\n\n", avg_cpu, std_cpu);
    fprintf(csvSingleThread, "%d,%.3f±%.3f,%.3f±%.3f\n", dim, avg_kernel, std_kernel, avg_cpu, std_cpu);

    // Clean up.
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

// Part 3: Kernel-Only Experiment (varying block sizes)
// Measures only the GPU kernel execution time (ignoring data transfers and CPU time).
// Kernel metrics are printed once per matrix dimension.
void runKernelOnlyExperiment(int dim, int blockSize, FILE* csvKernel) {
    size_t totalBytes = dim * dim * sizeof(float);

    // Allocate and initialize host matrices.
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

    // Allocate device memory.
    float* devA, * devB, * devC;
    cudaMalloc((void**)&devA, totalBytes);
    cudaMalloc((void**)&devB, totalBytes);
    cudaMalloc((void**)&devC, totalBytes);

    // Copy matrices to device.
    cudaMemcpy(devA, hostA, totalBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, hostB, totalBytes, cudaMemcpyHostToDevice);

    // Print kernel metrics once per matrix dimension.
    static int lastPrintedDimKernel = -1;
    if (lastPrintedDimKernel != dim) {
        int loadsPerThread = 2 * dim;
        int flopsPerThread = 2 * dim;
        float CGMA = (float)flopsPerThread / loadsPerThread;
        printf(">> [Parallel Kernel] For matrix dimension %d:\n", dim);
        printf("Each thread loads %d values and performs %d floating-point operations.\n", loadsPerThread, flopsPerThread);
        printf("CGMA Ratio per thread: %.2f\n\n", CGMA);
        lastPrintedDimKernel = dim;
    }

    // Set grid and block dimensions.
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 numBlocks((dim + blockSize - 1) / blockSize, (dim + blockSize - 1) / blockSize);

    float sum_kernel = 0;
    float sq_sum_kernel = 0;

    // Create CUDA events.
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    for (int iter = 0; iter < NUM_ITER; ++iter) {
        cudaEventRecord(startEvent, 0);
        gpuMatrixMultiply << <numBlocks, threadsPerBlock >> > (devA, devB, devC, dim);
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        float kernelTime = 0.0f;
        cudaEventElapsedTime(&kernelTime, startEvent, stopEvent);

        sum_kernel += kernelTime;
        sq_sum_kernel += kernelTime * kernelTime;
    }

    float avg_kernel = sum_kernel / NUM_ITER;
    float std_kernel = sqrt(sq_sum_kernel / NUM_ITER - avg_kernel * avg_kernel);

    printf("----- Kernel-Only: Matrix %dx%d, Block Width %d -----\n", dim, dim, blockSize);
    printf("GPU Kernel Execution: Avg = %.3f ms, StdDev = %.3f ms\n\n", avg_kernel, std_kernel);
    fprintf(csvKernel, "%d,%d,%.3f±%.3f\n", dim, blockSize, avg_kernel, std_kernel);

    // Clean up.
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    free(hostA);
    free(hostB);
    free(hostC_gpu);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

// ----------------- Main -----------------

int main(void) {
    // Define matrix sizes and block sizes.
    int matrixSizes[5] = { 256, 512, 1024, 2048, 4096 };
    int blockSizes[5] = { 2, 4, 8, 16, 32 };

    // -----------------
    // Data Transfer Experiment.
    // -----------------
    FILE* csvTransfer = fopen("data_transfer_times.csv", "w");
    if (!csvTransfer) {
        printf("Error opening CSV file for data transfer experiments.\n");
        return EXIT_FAILURE;
    }
    fprintf(csvTransfer, "Matrix_Size,HostToDevice,DeviceToHost\n");
    for (int i = 0; i < 5; ++i) {
        runDataTransferExperiment(matrixSizes[i], csvTransfer);
    }
    fclose(csvTransfer);

    // -----------------
    // Part 2: GPU vs CPU Comparison using Single-Thread Kernel.
    // -----------------
    FILE* csvSingleThread = fopen("gpu_single_thread_times.csv", "w");
    if (!csvSingleThread) {
        printf("Error opening CSV file for single-thread experiments.\n");
        return EXIT_FAILURE;
    }
    fprintf(csvSingleThread, "Matrix_Size,GPU_SingleThread_Time,CPU_Time\n");
    for (int i = 0; i < 5; ++i) {
        runSingleThreadExperiment(matrixSizes[i], csvSingleThread);
    }
    fclose(csvSingleThread);

    // -----------------
    // Part 3: Kernel-Only Experiments (Varying Block Sizes).
    // -----------------
    FILE* csvKernelTime = fopen("kernel_times.csv", "w");
    if (!csvKernelTime) {
        printf("Error opening CSV file for kernel-only experiments.\n");
        return EXIT_FAILURE;
    }
    fprintf(csvKernelTime, "Matrix_Size,Block_Size,Kernel_Time\n");
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            runKernelOnlyExperiment(matrixSizes[i], blockSizes[j], csvKernelTime);
        }
    }
    fclose(csvKernelTime);

    return 0;
}
