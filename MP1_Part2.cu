// Bradley Stephen | 19bbs2 | 20207842
// March 4th, 2025
// ELEC 374 - Machine Problem 1 - Part 2

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <time.h>

// using to account for small diffs in FP arthimiteic betweehn cpu and gpu
#define TOLERANCE 1e-3f

// GPU kernel, each thread computes one element of output 
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

// CPU version 
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

// Compare gpu and cpu matracies with given tolerance of 1e-3f
int compareMatrices(const float* mat1, const float* mat2, int dim, float tol) {
    int totalElements = dim * dim;
    for (int i = 0; i < totalElements; ++i) {
        if (fabs(mat1[i] - mat2[i]) > tol)
            return 0;
    }
    return 1;
}

/*Run one experiment with the specified matrix dimension and block size.
  Once working ill re run epxerince for all matrix an block variation in main
  and store results in CSV to plot in excel later.*/ 
void runExperiment(int dim, int blockSize, FILE* csvTransfer, FILE* csvKernel, FILE* csvCpuGpu) {
    size_t totalBytes = dim * dim * sizeof(float);

    // Allocate host mem
    float* hostA = (float*)malloc(totalBytes);
    float* hostB = (float*)malloc(totalBytes);
    float* hostC_gpu = (float*)malloc(totalBytes);
    float* hostC_cpu = (float*)malloc(totalBytes);
    if (!hostA || !hostB || !hostC_gpu || !hostC_cpu) {
        printf("Failed to allocate host memory for dimension %d.\n", dim);
        exit(EXIT_FAILURE);
    }

    // Init matrices with random vals
    srand((unsigned)time(NULL));
    for (int i = 0; i < dim * dim; ++i) {
        hostA[i] = (float)(rand() % 100) / 10.0f;
        hostB[i] = (float)(rand() % 100) / 10.0f;
    }

    // Allocate device mem
    float* devA, * devB, * devC;
    cudaMalloc((void**)&devA, totalBytes);
    cudaMalloc((void**)&devB, totalBytes);
    cudaMalloc((void**)&devC, totalBytes);

    // Create CUDA events for timing
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // Measure H2D transfer time
    cudaEventRecord(startEvent, 0);
    cudaMemcpy(devA, hostA, totalBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, hostB, totalBytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float h2dTime = 0.0f;
    cudaEventElapsedTime(&h2dTime, startEvent, stopEvent);

    // Set up grid and block dims
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 numBlocks((dim + blockSize - 1) / blockSize, (dim + blockSize - 1) / blockSize);

    // Measure GPU kernel EX time
    cudaEventRecord(startEvent, 0);
    gpuMatrixMultiply<<<numBlocks, threadsPerBlock >>>(devA, devB, devC, dim);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float kernelTime = 0.0f;
    cudaEventElapsedTime(&kernelTime, startEvent, stopEvent);

    // Measure D2H transfer time.
    cudaEventRecord(startEvent, 0);
    cudaMemcpy(hostC_gpu, devC, totalBytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float d2hTime = 0.0f;
    cudaEventElapsedTime(&d2hTime, startEvent, stopEvent);

    // Measure CPU matrix mul time
    clock_t cpuStart = clock();
    cpuMatrixMultiply(hostA, hostB, hostC_cpu, dim);
    clock_t cpuEnd = clock();
    float cpuTime = 1000.0f * (cpuEnd - cpuStart) / CLOCKS_PER_SEC;

    // Validate GPU result
    int testPassed = compareMatrices(hostC_gpu, hostC_cpu, dim, TOLERANCE);

    // Print to terminal for clarity
    printf("----- Matrix Dimension: %dx%d, Block Width: %d -----\n", dim, dim, blockSize);
    printf("Host-to-Device Transfer: %.3f ms\n", h2dTime);
    printf("GPU Kernel Execution: %.3f ms\n", kernelTime);
    printf("Device-to-Host Transfer: %.3f ms\n", d2hTime);
    printf("CPU Computation Time: %.3f ms\n", cpuTime);
    if (testPassed)
        printf("Test PASSED: GPU and CPU results are equivalent!\n\n");
    else
        printf("Test FAILED: GPU and CPU results do not match.\n\n");

    // Write timing data to CSV files.
    fprintf(csvTransfer, "%d,%.3f,%.3f\n", dim, h2dTime, d2hTime);
    fprintf(csvKernel, "%d,%d,%.3f\n", dim, blockSize, kernelTime);
    fprintf(csvCpuGpu, "%d,%.3f,%.3f\n", dim, cpuTime, kernelTime);

    // Clean up device and host resources.
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

int main(void) {
    // Define matrix/block sizes for testing
    int matrixSizes[5] = { 256, 512, 1024, 2048, 4096 };
    int blockSizes[5] = { 2, 4, 8, 16, 32 };

    // Open CSVs for writing the results and ploting later
    FILE* csvDataTrans = fopen("data_transfer_times.csv", "w");
    FILE* csvKernelTime = fopen("kernel_times.csv", "w");
    FILE* csvCpuVsGpu = fopen("cpu_vs_gpu_times.csv", "w");
    if (!csvDataTrans || !csvKernelTime || !csvCpuVsGpu) {
        printf("Error opening CSV files.\n");
        return EXIT_FAILURE;
    }

    // CSV Headerss.
    fprintf(csvDataTrans, "Matrix_Size,HostToDevice,DeviceToHost\n");
    fprintf(csvKernelTime, "Matrix_Size,Block_Size,Kernel_Time\n");
    fprintf(csvCpuVsGpu, "Matrix_Size,CPU_Time,GPU_Time\n");

    // Run experiments for all matrix/block configurations.
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            runExperiment(matrixSizes[i], blockSizes[j], csvDataTrans, csvKernelTime, csvCpuVsGpu);
        }
    }

    // Close CSVs.
    fclose(csvDataTrans);
    fclose(csvKernelTime);
    fclose(csvCpuVsGpu);

    return 0;
}
