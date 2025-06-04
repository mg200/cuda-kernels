#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>
#include <math.h>
#include <cmath>
#include "utils.h"
using namespace std;
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

__global__ void workInefficientScan(float *d_input, float *d_output, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= n) return;

    d_output[tid] = d_input[tid];

    for (int offset = 1; offset < n; offset *= 2) {
        __syncthreads();
        float val = 0;
        if (tid >= offset)
            val = d_output[tid - offset];
        __syncthreads();
        d_output[tid] += val;
    }
}
__global__ void scanBlockKernelWorkInefficient(float* d_input, float* d_output, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    float sum = 0;
    for (int i = 0; i <= gid; ++i) {
        sum += d_input[i];
    }
    d_output[gid] = sum;
}



int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <input_file> <output_file>" << endl;
        return 1;
    }

    FILE *f1 = fopen(argv[1], "r");
    if (f1 == nullptr) {
        cerr << "Error: Could not open input file " << argv[1] << endl;
        return 1;
    }

    int size;
    float *h_array = readArr<float>(f1, &size);
    fclose(f1);
    float *h_result = (float *)malloc(sizeof(float) * size);

    float *d_array, *d_result;
    cudaMalloc(&d_array, size * sizeof(float));
    cudaMalloc(&d_result, size * sizeof(float));
    cudaMemcpy(d_array, h_array, size * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 1024;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    // workInefficientScan<<<blocks, threadsPerBlock>>>(d_array, d_result, size);
    scanBlockKernelWorkInefficient<<<blocks, threadsPerBlock>>>(d_array, d_result, size);

    cudaDeviceSynchronize();

    cudaMemcpy(h_result, d_result, size * sizeof(float), cudaMemcpyDeviceToHost);

    FILE *of = fopen(argv[2], "w");
    for (int j = 0; j < size; ++j)
        fprintf(of, "%f ", h_result[j]);
    fprintf(of, "\n");
    fclose(of);
    printf("last element is: %f", h_result[size-1]);
    cudaFree(d_array);
    cudaFree(d_result);
    free(h_array);
    free(h_result);

    return 0;
}

/*
nvcc -o naive naive.cu
naive input.txt out_naive.txt
*/
