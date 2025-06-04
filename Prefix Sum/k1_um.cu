#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>
#include <math.h>
#include <cmath>
#include "utils.h"
using namespace std;

/* Prefix sum work-efficient implementation using CUDA unified memory */

#define FILTER_SIZE 25
#define CHANNELS 3
#define BLOCK_X 16
#define BLOCK_Y 16
#define cudaCheckError()                                                                 \
    {                                                                                    \
        cudaError_t e = cudaGetLastError();                                              \
        if (e != cudaSuccess)                                                            \
        {                                                                                \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    }

void prefix_sum(int *A, int *B, int size)
{
    B[0] = A[0];
    for (int i = 1; i < size; i++)
    {
        B[i] = B[i - 1] + A[i];
    }
}

double sum_all(double *A, int size)
{
    double x = 0;
    for (int i = 0; i < size; i++)
    {
        x += A[i];
    }
    return x;
}
__global__ void scanBlockKernel(double *d_input, double *d_output, double *d_block_sums, int n)
{
    extern __shared__ double temp[];
    int tid = threadIdx.x;
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= blockDim.x || gid >= n)
        return;

    // Inclusive scan: initialize with the current value (not shifted)
    temp[tid] = d_input[gid];
    __syncthreads();

    // Inclusive scan loop
    for (int offset = 1; offset < blockDim.x; offset *= 2)
    {
        double val = 0;
        if (tid >= offset)
            val = temp[tid - offset];
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }

    d_output[gid] = temp[tid];

    if (tid == blockDim.x - 1 && d_block_sums != nullptr)
        d_block_sums[blockIdx.x] = temp[tid];
}

__global__ void addOffsetsKernel(double *d_output, double *d_block_offsets, int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n || blockIdx.x == 0)
        return;

    d_output[gid] += d_block_offsets[blockIdx.x];
}

void scanBlockSums(double *d_block_sums, double *d_block_offsets, int blocks)
{
    if (blocks <= 1024)
    {
        int threadsPerBlock = 1024;
        int sharedMemSize = threadsPerBlock * sizeof(double);
        scanBlockKernel<<<1, blocks, sharedMemSize>>>(d_block_sums, d_block_offsets, nullptr, blocks);
    }
    else
    {
        double *d_recursive_sums, *d_recursive_offsets;
        int threadsPerBlock = 1024;
        int numBlocks = (blocks + threadsPerBlock - 1) / threadsPerBlock;

        cudaMalloc(&d_recursive_sums, numBlocks * sizeof(double));
        cudaMalloc(&d_recursive_offsets, numBlocks * sizeof(double));

        scanBlockKernel<<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(d_block_sums, d_block_offsets, d_recursive_sums, blocks);
        scanBlockSums(d_recursive_sums, d_recursive_offsets, numBlocks);
        addOffsetsKernel<<<numBlocks, threadsPerBlock>>>(d_block_offsets, d_recursive_offsets, blocks);

        cudaFree(d_recursive_sums);
        cudaFree(d_recursive_offsets);
    }
}
int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        cerr << "Usage: " << argv[0] << " <input_file> <output_file>" << endl;
        return 1;
    }

    FILE *f1 = fopen(argv[1], "r");
    if (f1 == nullptr)
    {
        cerr << "Error: Could not open input file " << argv[1] << endl;
        return 1;
    }

    int size;
    double *h_array = readArr<double>(f1, &size);
    // printf("h_array[%d] is: %f\n", 0, h_array[0]);
    fclose(f1);
    double sum = sum_all(h_array, size);
    printf("sum is : %f\n", sum);
    printf("size is %d\n", size);
    double *h_result = (double *)malloc(sizeof(double) * size);

    double *d_array, *d_result;
    cudaMallocManaged(&d_array, size * sizeof(double));
    cudaMallocManaged(&d_result, size * sizeof(double));
    memcpy(d_array, h_array, size * sizeof(double)); // Just a host memcpy

    int threadsPerBlock = 1024;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemSize = threadsPerBlock * sizeof(double);

    double *d_block_sums, *d_block_offsets;
    cudaMallocManaged(&d_block_sums, blocks * sizeof(double));
    cudaMallocManaged(&d_block_offsets, blocks * sizeof(double));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // GPU computation starts
    scanBlockKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(d_array, d_result, d_block_sums, size);
    scanBlockSums(d_block_sums, d_block_offsets, blocks);
    addOffsetsKernel<<<blocks, threadsPerBlock>>>(d_result, d_block_offsets, size);
    // GPU computation ends

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Measure time
    float gpu_time_ms = 0.0f;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    printf("Total GPU computation time: %.3f ms\n", gpu_time_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    ///

    cudaMemcpy(h_result, d_result, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize(); // Ensure computation is done before accessing from host

    FILE *of = fopen(argv[2], "w");
    if (of == nullptr)
    {
        cerr << "Error: Could not open output file " << argv[2] << endl;
        return 1;
    }
    // for (int j = size - 5; j < size; ++j)
    // {
    //     printf("h_result[%d] = %f\n", j, h_result[j]);
    // }

    for (int j = 0; j < size; ++j)
        fprintf(of, "%f ", h_result[j]);

    fprintf(of, "\n");
    printf("Written to file successfully!\n");

    fclose(of);
    printf("last element is: %f", h_result[size - 1]);
    free(h_array);
    free(h_result);
    cudaFree(d_array);
    cudaFree(d_result);
    cudaFree(d_block_sums);
    cudaFree(d_block_offsets);

    return 0;
}

/*
nvcc -o k1_um k1_um.cu
k1_um input.txt out_um.txt

*/