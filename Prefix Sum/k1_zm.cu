#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>
#include <math.h>
#include <cmath>
#include "utils.h"
using namespace std;

/* Prefix sum work-efficient implementation using CUDA zero-mapped memory */
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

    // initialize with the current value 
    temp[tid] = d_input[gid];
    __syncthreads();

    // inclusive scan loop
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

    // Store last value in block as block sum (if needed)
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
    fscanf(f1, "%d", &size);

    ///
    double *h_array, *h_result;
    double *d_array, *d_result;

    cudaHostAlloc((void **)&h_array, size * sizeof(double), cudaHostAllocMapped);
    cudaHostAlloc((void **)&h_result, size * sizeof(double), cudaHostAllocMapped);

    readArr(f1, h_array, size);
    fclose(f1);

    // Get device pointer to mapped memory
    cudaHostGetDevicePointer((void **)&d_array, (void *)h_array, 0);
    cudaHostGetDevicePointer((void **)&d_result, (void *)h_result, 0);

    ///

    double sum = sum_all(h_array, size);
    printf("sum is : %f\n", sum);
    printf("size is : %d\n", size);

    int threadsPerBlock = 1024;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemSize = threadsPerBlock * sizeof(double);

    double *d_block_sums, *d_block_offsets;
    cudaMalloc(&d_block_sums, blocks * sizeof(double));
    cudaMalloc(&d_block_offsets, blocks * sizeof(double));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start right before the first kernel launch
    cudaEventRecord(start);

    // GPU computation starts
    scanBlockKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(d_array, d_result, d_block_sums, size);
    scanBlockSums(d_block_sums, d_block_offsets, blocks);
    addOffsetsKernel<<<blocks, threadsPerBlock>>>(d_result, d_block_offsets, size);
    // GPU computation ends

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time_ms = 0.0f;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    printf("Total GPU computation time: %.3f ms\n", gpu_time_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    ///

    double *cpu_result = (double *)malloc(size * sizeof(double));
    cpu_result[0] = h_array[0];
    for (int i = 1; i < size; i++)
    {
        cpu_result[i] = cpu_result[i - 1] + h_array[i];
    }

    for (int i = 0; i < size; i++)
    {
        if (fabs(h_result[i] - cpu_result[i]) > 1e-6)
        {
            printf("Mismatch at %d: GPU=%f, CPU=%f\n", i, h_result[i], cpu_result[i]);
            break;
        }
    }
    free(cpu_result);

    FILE *of = fopen(argv[2], "w");
    if (of == nullptr)
    {
        cerr << "Error: Could not open output file " << argv[2] << endl;
        return 1;
    }

    for (int j = 0; j < size; ++j)
        fprintf(of, "%f ", h_result[j]);

    fprintf(of, "\n");
    printf("Written to file successfully!\n");

    fclose(of);
    printf("last element is: %f", h_result[size - 1]);
    cudaFreeHost(h_array);
    cudaFreeHost(h_result);
    cudaFree(d_block_sums);
    cudaFree(d_block_offsets);
    return 0;
}
/*
commands to compile and run:
nvcc -o k1_zm k1_zm.cu
k1_zm input.txt out_zm.txt

*/