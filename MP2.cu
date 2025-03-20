#include <cstdio>               // printf, fopen, etc.
#include <cstdlib>              // rand, srand
#include <cmath>                // fabs
#include <cuda.h>
#include <vector>               // std::vector
#include <string>               // std::string
#include <chrono>               // timing
#include <iostream>             // std::cout, std::cerr

//----------------------------------------------------------
// Tiled Matrix Multiplication Kernel
//----------------------------------------------------------
// Each thread computes one element of the output matrix P.
// Shared memory is used to reuse sub-blocks (tiles) of M and N.

__global__
void tiledMatMulKernel(const float* __restrict__ M,
    const float* __restrict__ N,
    float* __restrict__ P,
    int width,
    int TILE_WIDTH)
{
    // We will dynamically allocate shared memory:
    // half for M (TILE_WIDTH x TILE_WIDTH floats)
    // half for N (TILE_WIDTH x TILE_WIDTH floats)
    extern __shared__ float sharedMem[];
    float* Mds = sharedMem;
    float* Nds = sharedMem + TILE_WIDTH * TILE_WIDTH;

    // Thread indices
    int bx = blockIdx.x;   // block index in x-dim
    int by = blockIdx.y;   // block index in y-dim
    int tx = threadIdx.x;  // thread index in x-dim
    int ty = threadIdx.y;  // thread index in y-dim

    // Compute row, col of P to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;

    // Number of tiles needed to cover the entire matrix row/col
    int numTiles = width / TILE_WIDTH;  // assumes width % TILE_WIDTH == 0

    // Loop over sub-tiles
    for (int t = 0; t < numTiles; t++)
    {
        // Load data into shared memory
        Mds[ty * TILE_WIDTH + tx] = M[Row * width + (t * TILE_WIDTH + tx)];
        Nds[ty * TILE_WIDTH + tx] = N[(t * TILE_WIDTH + ty) * width + Col];

        __syncthreads();

        // Multiply the loaded tiles
        for (int k = 0; k < TILE_WIDTH; k++)
        {
            Pvalue += Mds[ty * TILE_WIDTH + k] * Nds[k * TILE_WIDTH + tx];
        }
        __syncthreads();
    }

    // Write result to global memory
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
bool compareResults(const float* ref, const float* gpu, int size, float tolerance = 1e-5f)
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
    // Title in terminal
    printf("============================================================\n");
    printf("   Machine Problem 2: Tiled Matrix Multiplication (GPU)   \n");
    printf("============================================================\n\n");

    // Define matrix sizes and tile widths to test
    std::vector<int> matrixSizes = { 256, 512, 1024, 2048, 4096 };
    std::vector<int> tileWidths = { 2, 4, 8, 16, 32 };

    // Open CSV file for results
    FILE* fp = fopen("MP2.csv", "w");
    if (!fp)
    {
        fprintf(stderr, "Error opening MP2.csv for writing.\n");
        return -1;
    }

    // Write CSV header
    // We'll store CPUTimeMs, GPUKernelTimeMs, HostToDeviceMs, DeviceToHostMs
    fprintf(fp, "MatrixSize,TileWidth,CPUTimeMs,GPUKernelTimeMs,HostToDeviceMs,DeviceToHostMs\n");

    // Loop over each matrix size
    for (int size : matrixSizes)
    {
        size_t bytes = static_cast<size_t>(size) * size * sizeof(float);

        // Allocate host memory
        float* h_M = (float*)malloc(bytes);
        float* h_N = (float*)malloc(bytes);
        float* h_P = (float*)malloc(bytes);  // GPU result
        float* h_ref = (float*)malloc(bytes);  // CPU reference

        // Initialize host matrices (random)
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

        // Print CPU reference time once per matrix size
        printf("------------------------------------------------------------\n");
        printf("Matrix Size: %d x %d\n", size, size);
        printf("CPU Reference Time: %.5f ms\n", cpuMs);
        printf("------------------------------------------------------------\n");

        // Loop over each tile width
        for (int TW : tileWidths)
        {
            // Skip if size is not divisible by TW
            if (size % TW != 0)
            {
                printf("  [TileWidth=%d] Skipped (matrix not divisible by tile width)\n", TW);
                continue;
            }

            // Allocate device memory
            float* d_M, * d_N, * d_P;
            cudaMalloc((void**)&d_M, bytes);
            cudaMalloc((void**)&d_N, bytes);
            cudaMalloc((void**)&d_P, bytes);

            // Measure Host->Device transfer
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

            // Kernel launch parameters
            dim3 dimBlock(TW, TW);
            dim3 dimGrid(size / TW, size / TW);

            // Shared memory = 2 * (TILE_WIDTH * TILE_WIDTH * sizeof(float))
            size_t sharedMemSize = 2 * TW * TW * sizeof(float);

            // Measure kernel execution time (only the kernel!)
            cudaEvent_t kStart, kStop;
            cudaEventCreate(&kStart);
            cudaEventCreate(&kStop);
            cudaEventRecord(kStart);

            tiledMatMulKernel << <dimGrid, dimBlock, sharedMemSize >> > (d_M, d_N, d_P, size, TW);

            cudaEventRecord(kStop);
            cudaEventSynchronize(kStop);
            float kernelMs = 0.0f;
            cudaEventElapsedTime(&kernelMs, kStart, kStop);

            // Measure Device->Host transfer
            cudaEvent_t d2hStart, d2hStop;
            cudaEventCreate(&d2hStart);
            cudaEventCreate(&d2hStop);
            cudaEventRecord(d2hStart);

            cudaMemcpy(h_P, d_P, bytes, cudaMemcpyDeviceToHost);

            cudaEventRecord(d2hStop);
            cudaEventSynchronize(d2hStop);
            float d2hMs = 0.0f;
            cudaEventElapsedTime(&d2hMs, d2hStart, d2hStop);

            // Verify correctness
            bool pass = compareResults(h_ref, h_P, size * size);

            // Print per-tile summary in terminal
            // GPU Kernel time only: 'kernelMs'
            // H->D and D->H are separate, not added to kernel time
            printf("  [TileWidth=%2d] KernelTime: %.5f ms | H->D: %.5f ms | D->H: %.5f ms | ",
                TW, kernelMs, h2dMs, d2hMs);
            if (pass) printf("Result: PASSED\n");
            else     printf("Result: FAILED\n");

            // Write to CSV
            fprintf(fp, "%d,%d,%.5f,%.5f,%.5f,%.5f\n",
                size, TW, cpuMs, kernelMs, h2dMs, d2hMs);

            // Cleanup device
            cudaFree(d_M);
            cudaFree(d_N);
            cudaFree(d_P);
            cudaEventDestroy(h2dStart);
            cudaEventDestroy(h2dStop);
            cudaEventDestroy(kStart);
            cudaEventDestroy(kStop);
            cudaEventDestroy(d2hStart);
            cudaEventDestroy(d2hStop);
        }

        // Free host memory
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
