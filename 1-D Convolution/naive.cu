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

/*
    1D Convolution kernel
    array: input array
    n: size of the array
    mask: convolution mask
    m: size of the mask
    result: output array
*/

__global__ void convolution_1d_kernel(float* array, float* mask, float* result, int n, int m) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // global thread index

    int radius = m / 2; // radius of the mask

    int start = tid - radius; // start index of the mask

    int temp = 0;

    for (int j = 0; j < m; j++) {
        if (start + j >= 0 && start + j < n) {
            temp += array[start + j] * mask[j];
        }
    }

    if (tid < n) {
        result[tid] = temp;
    }

}

void verify_results(float* array, float* mask, float* result, int n, int m) {
    int radius = m / 2;
    int start;
    for (int i = 0; i < n; i++) {
        int temp = 0;
        start = i - radius;
        for (int j = 0; j < m; j++) {
            if (start + j >= 0 && start + j < n) {
                temp += array[start + j] * mask[j];
            }
        }
        printf("%d %f\n", temp, result[i]);
        assert(temp == result[i]);
    }
}

void convolve_1d_Device(float *d_array, float *d_mask, float *d_result, int n, int m) {
    dim3 DimBlock(256, 1, 1);
    dim3 DimGrid((DimBlock.x + n - 1) / DimBlock.x, 1, 1);
    convolution_1d_kernel<<<DimGrid, DimBlock>>>(d_array, d_mask, d_result, n, m);
    gpuErrchk(cudaPeekAtLastError());
}

void convolve_1d(float *h_array, float *h_mask, float *h_result, int n, int m) {
    
    float *d_array, *d_mask, *d_result;
    cudaMalloc(&d_array, n * sizeof(float));
    cudaMalloc(&d_mask, m * sizeof(float));
    cudaMalloc(&d_result, n * sizeof(float));


    cudaMemcpy(d_array, h_array, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, m * sizeof(float), cudaMemcpyHostToDevice);

    // call the kernel
    convolve_1d_Device(d_array, d_mask, d_result, n, m);
    
    cudaMemcpy(h_result, d_result, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_array);
    cudaFree(d_mask);
    cudaFree(d_result);
}



int main(int argc, char* argv[]) {

    // // Randomize inputs
    // int n = 1 << 20;
    // int m = 7;
    // float *h_array = (float*)malloc(n * sizeof(float));
    // float *h_mask = (float*)malloc(m * sizeof(float));
    // float *h_result = (float*)malloc(n * sizeof(float));

    // for (int i = 0; i < n; i++) { h_array[i] = rand() % 100; }

    // for (int i = 0; i < m; i++) { h_mask[i] = rand() % 10; }
    // // End Randomization


    
    // File input
    int n, m;
    float *h_array, *h_mask, *h_result;

    char *in_filename = argv[1];
    char *mask_filename = argv[2];
    char *out_filename = argv[3];

    FILE* file = fopen(in_filename, "r");
    if (!file) { fprintf(stderr, "Failed to open input file"); }
    fscanf(file, "%d", &n);
    h_array = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) fscanf(file, "%f", &h_array[i]);

    file = fopen(mask_filename, "r");
    if (!file) { fprintf(stderr, "Failed to open mask file"); }
    fscanf(file, "%d", &m);
    h_mask = (float*)malloc(m * sizeof(float));
    for (int i = 0; i < m; i++) fscanf(file, "%f", &h_mask[i]);
    // End file input
    

    h_result = (float*)malloc(n * sizeof(float));

    convolve_1d(h_array, h_mask, h_result, n, m);
    
    verify_results(h_array, h_mask, h_result, n, m);

    printf("Results match!\n");
    // Write output to file


    file = fopen(out_filename, "w");
    if (!file) { fprintf(stderr, "Failed to open output file"); }
    for (int i = 0; i < n; i++) fprintf(file, "%f ", h_result[i]);
    

    // free memory
    free(h_array);
    free(h_mask);
    free(h_result);

    return 0;
}


/*
What can we optimize?
- We are not modifying the mask, so we can use constant memory/cache (inside the GPU) for the mask, which is faster than global memory

- For a mask of size 3, thread 0 accesses elements 0 and 1, thread 1 accesses elements 0, 1, and 2. 
    There is a lot of overlap, and hance locality. So, we can use shared memory to store the mask and avoid redundant accesses to global memory

*/