#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>
#include <chrono>
#include <iostream>

#define NUM_ITER 10
#define TOLERANCE 1e-3f

// Tile dimensions for the output matrix P.
#define TILE_ROW 12      // Number of rows per tile (block)
#define TILE_COL 18      // Number of columns per tile (block)
// Tile size in the common dimension (the “depth” of the tile)
#define TILE_COMMON 18

//---------------------------------------------------------------------
// Revised Tiled Matrix Multiplication Kernel with Boundary Checks
//
// Computes: P = M * N, where M is A_rows x A_cols and N is A_cols x B_cols.
// Each block computes a TILE_ROW x TILE_COL tile of P using shared memory.
//---------------------------------------------------------------------
__global__
void tiledMatMulKernelBoundary(const float* __restrict__ M,
                               const float* __restrict__ N,
                               float* __restrict__ P,
                               int A_rows, int A_cols, int B_cols)
{
    // Dynamically allocated shared memory:
    // tileM: TILE_ROW x TILE_COMMON, tileN: TILE_COMMON x TILE_COL
    extern __shared__ float shared[];
    float* tileM = shared;                      // size: TILE_ROW * TILE_COMMON
    float* tileN = shared + TILE_ROW * TILE_COMMON;  // size: TILE_COMMON * TILE_COL

    int row = blockIdx.y * TILE_ROW + threadIdx.y;
    int col = blockIdx.x * TILE_COL + threadIdx.x;
    float Pvalue = 0.0f;

    // Number of tiles along the common dimension
    int numTiles = (A_cols + TILE_COMMON - 1) / TILE_COMMON;

    for (int t = 0; t < numTiles; t++)
    {
        // Load tileM from matrix M with boundary check.
        int m_col = t * TILE_COMMON + threadIdx.x;
        if (row < A_rows && m_col < A_cols)
            tileM[threadIdx.y * TILE_COMMON + threadIdx.x] = M[row * A_cols + m_col];
        else
            tileM[threadIdx.y * TILE_COMMON + threadIdx.x] = 0.0f;

        // Load tileN from matrix N with boundary check.
        for (int i = threadIdx.y; i < TILE_COMMON; i += TILE_ROW)
        {
            int n_row = t * TILE_COMMON + i;
            if (n_row < A_cols && col < B_cols)
                tileN[i * TILE_COL + threadIdx.x] = N[n_row * B_cols + col];
            else
                tileN[i * TILE_COL + threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Multiply the two tiles.
        for (int k = 0; k < TILE_COMMON; k++)
        {
            Pvalue += tileM[threadIdx.y * TILE_COMMON + k] * tileN[k * TILE_COL + threadIdx.x];
        }
        __syncthreads();
    }

    // Write the result with boundary check.
    if (row < A_rows && col < B_cols)
        P[row * B_cols + col] = Pvalue;
}

//---------------------------------------------------------------------
// CPU Reference Multiplication for Non-Square Matrices
//---------------------------------------------------------------------
void cpuMatrixMulBoundary(const float* M, const float* N, float* P,
                          int A_rows, int A_cols, int B_cols)
{
    for (int i = 0; i < A_rows; i++)
    {
        for (int j = 0; j < B_cols; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < A_cols; k++)
            {
                sum += M[i * A_cols + k] * N[k * B_cols + j];
            }
            P[i * B_cols + j] = sum;
        }
    }
}

//---------------------------------------------------------------------
// Compare Results (CPU vs. GPU)
//---------------------------------------------------------------------
bool compareResultsBoundary(const float* ref, const float* gpu, int size, float tolerance = TOLERANCE)
{
    for (int i = 0; i < size; i++)
    {
        if (fabs(ref[i] - gpu[i]) > tolerance)
            return false;
    }
    return true;
}

//---------------------------------------------------------------------
// Main function for the Bonus Part (with CSV output)
//---------------------------------------------------------------------
int main()
{
    // Open CSV file for results.
    FILE* fp = fopen("MP2Bonus.csv", "w");
    if (!fp)
    {
        std::cerr << "Error opening MP2Bonus.csv for writing." << std::endl;
        return -1;
    }
    // Write CSV header.
    fprintf(fp, "Test,A_rows,A_cols,B_cols,CPUTimeMs,GPUKernelAvgMs,GPUKernelStdMs,Result\n");

    // Define two test cases:
    // Test 1: M: 750 x 800, N: 800 x 850 => P: 750 x 850
    // Test 2: M: 2000 x 1750, N: 1750 x 1900 => P: 2000 x 1900
    struct TestCase {
        int A_rows, A_cols, B_cols;
    };

    TestCase tests[2] = { {750, 800, 850}, {2000, 1750, 1900} };

    for (int test = 0; test < 2; test++)
    {
        int A_rows = tests[test].A_rows;
        int A_cols = tests[test].A_cols;
        int B_cols = tests[test].B_cols;
        size_t sizeM = A_rows * A_cols * sizeof(float);
        size_t sizeN = A_cols * B_cols * sizeof(float);
        size_t sizeP = A_rows * B_cols * sizeof(float);

        float* h_M = (float*)malloc(sizeM);
        float* h_N = (float*)malloc(sizeN);
        float* h_P = (float*)malloc(sizeP);
        float* h_ref = (float*)malloc(sizeP);

        // Initialize matrices randomly.
        srand(0);
        for (int i = 0; i < A_rows * A_cols; i++)
        {
            h_M[i] = static_cast<float>(rand() % 100) / 10.0f;
        }
        for (int i = 0; i < A_cols * B_cols; i++)
        {
            h_N[i] = static_cast<float>(rand() % 100) / 10.0f;
        }

        // CPU reference multiplication.
        auto cpuStart = std::chrono::high_resolution_clock::now();
        cpuMatrixMulBoundary(h_M, h_N, h_ref, A_rows, A_cols, B_cols);
        auto cpuEnd = std::chrono::high_resolution_clock::now();
        double cpuTime = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
        std::cout << "Test " << test+1 << ": CPU Reference Time: " << cpuTime << " ms" << std::endl;

        // Allocate device memory.
        float *d_M, *d_N, *d_P;
        cudaMalloc((void**)&d_M, sizeM);
        cudaMalloc((void**)&d_N, sizeN);
        cudaMalloc((void**)&d_P, sizeP);

        // Copy input matrices to device.
        cudaMemcpy(d_M, h_M, sizeM, cudaMemcpyHostToDevice);
        cudaMemcpy(d_N, h_N, sizeN, cudaMemcpyHostToDevice);

        // Compute grid dimensions (ceiling division).
        dim3 dimBlock(TILE_COL, TILE_ROW);  // 18 x 12 threads per block.
        dim3 dimGrid((B_cols + TILE_COL - 1) / TILE_COL, (A_rows + TILE_ROW - 1) / TILE_ROW);
        // Total shared memory size.
        size_t sharedMemSize = (TILE_ROW * TILE_COMMON + TILE_COMMON * TILE_COL) * sizeof(float);

        // Run the kernel NUM_ITER times to compute average and standard deviation.
        double sumKernel = 0.0, sumSqKernel = 0.0;
        for (int iter = 0; iter < NUM_ITER; iter++)
        {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            tiledMatMulKernelBoundary<<<dimGrid, dimBlock, sharedMemSize>>>(d_M, d_N, d_P, A_rows, A_cols, B_cols);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float kernelTime = 0.0f;
            cudaEventElapsedTime(&kernelTime, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            sumKernel += kernelTime;
            sumSqKernel += kernelTime * kernelTime;
        }
        double avgKernel = sumKernel / NUM_ITER;
        double stdKernel = sqrt((sumSqKernel / NUM_ITER) - (avgKernel * avgKernel));
        std::cout << "Test " << test+1 << ": GPU Kernel Time: Avg = " << avgKernel << " ms, Std = " << stdKernel << " ms" << std::endl;

        // Copy result back from device.
        cudaMemcpy(h_P, d_P, sizeP, cudaMemcpyDeviceToHost);

        // Verify the result.
        bool correct = compareResultsBoundary(h_ref, h_P, A_rows * B_cols);
        std::cout << "Test " << test+1 << ": Result " << (correct ? "PASSED" : "FAILED") << std::endl;
        std::cout << "---------------------------------------------" << std::endl;

        // Write results to CSV.
        fprintf(fp, "%d,%d,%d,%d,%.5f,%.5f,%.5f,%s\n", test+1, A_rows, A_cols, B_cols, cpuTime, avgKernel, stdKernel, (correct ? "PASSED" : "FAILED"));

        // Free device and host memory.
        cudaFree(d_M);
        cudaFree(d_N);
        cudaFree(d_P);
        free(h_M);
        free(h_N);
        free(h_P);
        free(h_ref);
    }

    fclose(fp);
    std::cout << "Results written to MP2Bonus.csv" << std::endl;
    return 0;
}
