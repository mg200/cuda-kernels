#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define TILE_SIZE 256
#define RADIUS 8 // Max assumed radius for padding

// Declare constant memory for the mask (improves global memory latency)
__constant__ float d_mask[16];

/*
    Optimized 1D Convolution Kernel with Input Tiling and Shared Memory
*/
__global__ void convolution_1d_kernel(const float* __restrict__ array, float* __restrict__ result, int n, int m) {
    __shared__ float shared_array[TILE_SIZE + 2 * RADIUS]; // Padding to reduce bank conflicts
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int radius = m / 2;
    int shared_index = tid + radius;

    // Load data into shared memory with padding
    if (gid < n) {
        shared_array[shared_index] = array[gid];
    } else {
        shared_array[shared_index] = 0.0f; // Padding for out-of-bounds threads
    }

    if (tid < radius) {
        shared_array[tid] = (gid - radius >= 0) ? array[gid - radius] : 0.0f;
        shared_array[shared_index + blockDim.x] = (gid + blockDim.x < n) ? array[gid + blockDim.x] : 0.0f;
    }

    __syncthreads();

    // Compute convolution
    float temp = 0.0f;
    #pragma unroll
    for (int j = 0; j < m; j++) {
        temp += shared_array[tid + j] * d_mask[j]; // Use constant memory for mask
    }

    if (gid < n) {
        result[gid] = temp;
    }
}

void convolve_1d_Device(float *d_array, float *d_result, int n, int m) {
    dim3 DimBlock(TILE_SIZE, 1, 1);
    dim3 DimGrid((n + TILE_SIZE - 1) / TILE_SIZE, 1, 1);
    convolution_1d_kernel<<<DimGrid, DimBlock>>>(d_array, d_result, n, m);
    gpuErrchk(cudaPeekAtLastError());
}

void convolve_1d(float *h_array, float *h_mask, float *h_result, int n, int m) {
    float *d_array, *d_result;
    
    // Allocate device memory
    cudaMalloc(&d_array, n * sizeof(float));
    cudaMalloc(&d_result, n * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_array, h_array, n * sizeof(float), cudaMemcpyHostToDevice);

    // Copy mask to constant memory (faster access)
    cudaMemcpyToSymbol(d_mask, h_mask, m * sizeof(float));

    convolve_1d_Device(d_array, d_result, n, m);

    // Copy result back to host
    cudaMemcpy(h_result, d_result, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_array);
    cudaFree(d_result);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input file> <mask file> <output file>\n", argv[0]);
        return 1;
    }

    int n, m;
    float *h_array, *h_mask, *h_result;
    
    FILE* file = fopen(argv[1], "r");
    if (!file) { fprintf(stderr, "Failed to open input file\n"); return 1; }
    fscanf(file, "%d", &n);
    h_array = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) fscanf(file, "%f", &h_array[i]);
    fclose(file);

    file = fopen(argv[2], "r");
    if (!file) { fprintf(stderr, "Failed to open mask file\n"); return 1; }
    fscanf(file, "%d", &m);
    h_mask = (float*)malloc(m * sizeof(float));
    for (int i = 0; i < m; i++) fscanf(file, "%f", &h_mask[i]);
    fclose(file);

    h_result = (float*)malloc(n * sizeof(float));

    convolve_1d(h_array, h_mask, h_result, n, m);

    file = fopen(argv[3], "w");
    if (!file) { fprintf(stderr, "Failed to open output file\n"); return 1; }
    for (int i = 0; i < n; i++) fprintf(file, "%f ", h_result[i]);
    fclose(file);

    free(h_array);
    free(h_mask);
    free(h_result);

    return 0;
}

// #include <stdio.h>
// #include <stdlib.h>
// #include <cuda.h>
// #include <assert.h>

// #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
// inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
// {
//    if (code != cudaSuccess)
//    {
//       fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
//       if (abort) exit(code);
//    }
// }

// #define TILE_SIZE 256

// /*
//     1D Convolution kernel with input tiling
//     array: input array
//     n: size of the array
//     mask: convolution mask
//     m: size of the mask
//     result: output array
// */

// __global__ void convolution_1d_kernel(float* array, float* mask, float* result, int n, int m) {
//     __shared__ float shared_array[TILE_SIZE + 16]; // Extra space for halo region
    
//     int tid = threadIdx.x;
//     int gid = blockIdx.x * blockDim.x + threadIdx.x;
//     int radius = m / 2;
//     int shared_index = tid + radius;
    
//     // Load data into shared memory
//     if (gid < n) {
//         shared_array[shared_index] = array[gid];
//     } else {
//         shared_array[shared_index] = 0.0f;
//     }
    
//     if (tid < radius) {
//         shared_array[tid] = (gid - radius >= 0) ? array[gid - radius] : 0.0f;
//         shared_array[shared_index + blockDim.x] = (gid + blockDim.x < n) ? array[gid + blockDim.x] : 0.0f;
//     }
    
//     __syncthreads();
    
//     // Compute convolution
//     float temp = 0.0f;
//     for (int j = 0; j < m; j++) {
//         temp += shared_array[tid + j] * mask[j];
//     }
    
//     if (gid < n) {
//         result[gid] = temp;
//     }
// }

// void convolve_1d_Device(float *d_array, float *d_mask, float *d_result, int n, int m) {
//     dim3 DimBlock(TILE_SIZE, 1, 1);
//     dim3 DimGrid((n + TILE_SIZE - 1) / TILE_SIZE, 1, 1);
//     convolution_1d_kernel<<<DimGrid, DimBlock>>>(d_array, d_mask, d_result, n, m);
//     gpuErrchk(cudaPeekAtLastError());
// }

// void convolve_1d(float *h_array, float *h_mask, float *h_result, int n, int m) {
//     float *d_array, *d_mask, *d_result;
//     cudaMalloc(&d_array, n * sizeof(float));
//     cudaMalloc(&d_mask, m * sizeof(float));
//     cudaMalloc(&d_result, n * sizeof(float));

//     cudaMemcpy(d_array, h_array, n * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_mask, h_mask, m * sizeof(float), cudaMemcpyHostToDevice);

//     convolve_1d_Device(d_array, d_mask, d_result, n, m);
    
//     cudaMemcpy(h_result, d_result, n * sizeof(float), cudaMemcpyDeviceToHost);

//     cudaFree(d_array);
//     cudaFree(d_mask);
//     cudaFree(d_result);
// }

// int main(int argc, char* argv[]) {
//     if (argc != 4) {
//         fprintf(stderr, "Usage: %s <input file> <mask file> <output file>\n", argv[0]);
//         return 1;
//     }

//     int n, m;
//     float *h_array, *h_mask, *h_result;
    
//     FILE* file = fopen(argv[1], "r");
//     if (!file) { fprintf(stderr, "Failed to open input file\n"); return 1; }
//     fscanf(file, "%d", &n);
//     h_array = (float*)malloc(n * sizeof(float));
//     for (int i = 0; i < n; i++) fscanf(file, "%f", &h_array[i]);
//     fclose(file);

//     file = fopen(argv[2], "r");
//     if (!file) { fprintf(stderr, "Failed to open mask file\n"); return 1; }
//     fscanf(file, "%d", &m);
//     h_mask = (float*)malloc(m * sizeof(float));
//     for (int i = 0; i < m; i++) fscanf(file, "%f", &h_mask[i]);
//     fclose(file);

//     h_result = (float*)malloc(n * sizeof(float));

//     convolve_1d(h_array, h_mask, h_result, n, m);

//     file = fopen(argv[3], "w");
//     if (!file) { fprintf(stderr, "Failed to open output file\n"); return 1; }
//     for (int i = 0; i < n; i++) fprintf(file, "%f ", h_result[i]);
//     fclose(file);

//     free(h_array);
//     free(h_mask);
//     free(h_result);

//     return 0;
// }
