#include <cstdio>               // printf, fopen, etc.
#include <cstdlib>              // rand, srand
#include <cmath>                // fabs, sqrt
#include <cuda.h>
#include <vector>               // std::vector
#include <string>               // std::string
#include <chrono>               // timing
#include <iostream>             // std::cout, std::cerr

#define NUM_ITER 10
#define TOLERANCE 1e-3f

//----------------------------------------------------------
// Tiled Matrix Multiplication Kernel
//----------------------------------------------------------
__global__
void tiledMatMulKernel(const float* __restrict__ M,
    const float* __restrict__ N,
    float* __restrict__ P,
    int width,
    int TILE_WIDTH)
{
    extern __shared__ float sharedMem[];
    float* Mds = sharedMem;
    float* Nds = sharedMem + TILE_WIDTH * TILE_WIDTH;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;
    int numTiles = width / TILE_WIDTH;  // assumes width % TILE_WIDTH == 0

    for (int t = 0; t < numTiles; t++)
    {
        Mds[ty * TILE_WIDTH + tx] = M[Row * width + (t * TILE_WIDTH + tx)];
        Nds[ty * TILE_WIDTH + tx] = N[(t * TILE_WIDTH + ty) * width + Col];
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; k++)
        {
            Pvalue += Mds[ty * TILE_WIDTH + k] * Nds[k * TILE_WIDTH + tx];
        }
        __syncthreads();
    }
    P[Row * width + Col] = Pvalue;
}

//----------------------------------------------------------
// CPU Reference Multiplication (for verification)
//----------------------------------------------------------
void cpuMatrixMul(const float* M, const float* N, float* P, int width)
{
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < width; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < width; k++)
            {
                sum += M[i * width + k] * N[k * width + j];
            }
            P[i * width + j] = sum;
        }
    }
}

//----------------------------------------------------------
// Compare GPU result with CPU result
//----------------------------------------------------------
bool compareResults(const float* ref, const float* gpu, int size, float tolerance = TOLERANCE)
{
    for (int i = 0; i < size; i++)
    {
        float diff = fabs(ref[i] - gpu[i]);
        if (diff > tolerance)
        {
            return false;
        }
    }
    return true;
}

//----------------------------------------------------------
// Main
//----------------------------------------------------------
int main()
{
    printf("============================================================\n");
    printf("   Machine Problem 2: Tiled Matrix Multiplication (GPU)   \n");
    printf("============================================================\n\n");

    std::vector<int> matrixSizes = { 256, 512, 1024, 2048, 4096 };
    std::vector<int> tileWidths = { 2, 4, 8, 16, 32 };

    // Open CSV file for results with extended columns for standard deviations
    FILE* fp = fopen("MP2.csv", "w");
    if (!fp)
    {
        fprintf(stderr, "Error opening MP2.csv for writing.\n");
        return -1;
    }
    fprintf(fp, "MatrixSize,TileWidth,CPUTimeMs,GPUKernelTimeAvgMs,GPUKernelTimeStdMs,HostToDeviceAvgMs,HostToDeviceStdMs,DeviceToHostAvgMs,DeviceToHostStdMs\n");

    // Loop over each matrix size
    for (int size : matrixSizes)
    {
        size_t bytes = static_cast<size_t>(size) * size * sizeof(float);
        float* h_M = (float*)malloc(bytes);
        float* h_N = (float*)malloc(bytes);
        float* h_P = (float*)malloc(bytes);   // GPU result
        float* h_ref = (float*)malloc(bytes); // CPU reference

        srand(0); // fixed seed for reproducibility
        for (int i = 0; i < size * size; i++)
        {
            h_M[i] = static_cast<float>(rand() % 100) / 10.0f;
            h_N[i] = static_cast<float>(rand() % 100) / 10.0f;
        }

        // Compute CPU reference and measure CPU time
        auto cpuStart = std::chrono::high_resolution_clock::now();
        cpuMatrixMul(h_M, h_N, h_ref, size);
        auto cpuEnd = std::chrono::high_resolution_clock::now();
        double cpuMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

        printf("------------------------------------------------------------\n");
        printf("Matrix Size: %d x %d\n", size, size);
        printf("CPU Reference Time: %.5f ms\n", cpuMs);
        printf("------------------------------------------------------------\n");

        // Loop over each tile width
        for (int TW : tileWidths)
        {
            if (size % TW != 0)
            {
                printf("  [TileWidth=%d] Skipped (matrix not divisible by tile width)\n", TW);
                continue;
            }

            // Allocate device memory once for this configuration
            float *d_M, *d_N, *d_P;
            cudaMalloc((void**)&d_M, bytes);
            cudaMalloc((void**)&d_N, bytes);
            cudaMalloc((void**)&d_P, bytes);

            // Setup kernel launch parameters
            dim3 dimBlock(TW, TW);
            dim3 dimGrid(size / TW, size / TW);
            size_t sharedMemSize = 2 * TW * TW * sizeof(float);

            double sumKernel = 0.0, sumH2d = 0.0, sumD2h = 0.0;
            double sumSqKernel = 0.0, sumSqH2d = 0.0, sumSqD2h = 0.0;

            // Run NUM_ITER iterations for timing
            for (int iter = 0; iter < NUM_ITER; iter++)
            {
                // Host-to-Device transfer timing
                cudaEvent_t h2dStart, h2dStop;
                cudaEventCreate(&h2dStart);
                cudaEventCreate(&h2dStop);
                cudaEventRecord(h2dStart);
                cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice);
                cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice);
                cudaEventRecord(h2dStop);
                cudaEventSynchronize(h2dStop);
                float h2dMs = 0.0f;
                cudaEventElapsedTime(&h2dMs, h2dStart, h2dStop);
                cudaEventDestroy(h2dStart);
                cudaEventDestroy(h2dStop);

                // Kernel execution timing
                cudaEvent_t kStart, kStop;
                cudaEventCreate(&kStart);
                cudaEventCreate(&kStop);
                cudaEventRecord(kStart);
                tiledMatMulKernel<<<dimGrid, dimBlock, sharedMemSize>>>(d_M, d_N, d_P, size, TW);
                cudaEventRecord(kStop);
                cudaEventSynchronize(kStop);
                float kernelMs = 0.0f;
                cudaEventElapsedTime(&kernelMs, kStart, kStop);
                cudaEventDestroy(kStart);
                cudaEventDestroy(kStop);

                // Device-to-Host transfer timing
                cudaEvent_t d2hStart, d2hStop;
                cudaEventCreate(&d2hStart);
                cudaEventCreate(&d2hStop);
                cudaEventRecord(d2hStart);
                cudaMemcpy(h_P, d_P, bytes, cudaMemcpyDeviceToHost);
                cudaEventRecord(d2hStop);
                cudaEventSynchronize(d2hStop);
                float d2hMs = 0.0f;
                cudaEventElapsedTime(&d2hMs, d2hStart, d2hStop);
                cudaEventDestroy(d2hStart);
                cudaEventDestroy(d2hStop);

                sumH2d   += h2dMs;
                sumKernel += kernelMs;
                sumD2h   += d2hMs;
                sumSqH2d   += h2dMs * h2dMs;
                sumSqKernel += kernelMs * kernelMs;
                sumSqD2h   += d2hMs * d2hMs;
            }

            // Compute averages
            double avgH2d   = sumH2d / NUM_ITER;
            double avgKernel = sumKernel / NUM_ITER;
            double avgD2h   = sumD2h / NUM_ITER;

            // Compute standard deviations
            double stdH2d   = sqrt((sumSqH2d / NUM_ITER) - (avgH2d * avgH2d));
            double stdKernel = sqrt((sumSqKernel / NUM_ITER) - (avgKernel * avgKernel));
            double stdD2h   = sqrt((sumSqD2h / NUM_ITER) - (avgD2h * avgD2h));

            // Verify correctness using result from last iteration
            bool pass = compareResults(h_ref, h_P, size * size);
            printf("  [TileWidth=%2d] Kernel: Avg = %.5f ms (Std = %.5f ms) | H->D: Avg = %.5f ms (Std = %.5f ms) | D->H: Avg = %.5f ms (Std = %.5f ms) | ",
                TW, avgKernel, stdKernel, avgH2d, stdH2d, avgD2h, stdD2h);
            if (pass) printf("Result: PASSED\n");
            else     printf("Result: FAILED\n");

            // Write averaged results to CSV
            fprintf(fp, "%d,%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n",
                size, TW, cpuMs, avgKernel, stdKernel, avgH2d, stdH2d, avgD2h, stdD2h);

            // Cleanup device memory for this configuration
            cudaFree(d_M);
            cudaFree(d_N);
            cudaFree(d_P);
        }

        free(h_M);
        free(h_N);
        free(h_P);
        free(h_ref);

        printf("\n");
    }

    fclose(fp);
    printf("============================================================\n");
    printf("All results written to MP2.csv\n");
    printf("============================================================\n");
    return 0;
}
