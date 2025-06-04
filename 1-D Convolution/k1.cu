#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>

#define FILTER_SIZE 3

// GPU Kernel for 1D Convolution
__global__ void conv1D_GPU(const float* input, const float* filter, float* output, int input_size, int filter_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < input_size) {
        float sum = 0.0f;
        for (int j = 0; j < filter_size; ++j) {
            int input_idx = idx + j - filter_size / 2;
            if (input_idx >= 0 && input_idx < input_size) {
                sum += input[input_idx] * filter[j];
            }
        }
        output[idx] = sum;
    }
}

void readData(const std::string& filename, std::vector<float>& data) {
    std::ifstream file(filename);
    int size;
    file >> size;
    data.resize(size);
    for (int i = 0; i < size; ++i) {
        file >> data[i];
    }
    file.close();
}

void writeData(const std::string& filename, const std::vector<float>& data) {
    std::ofstream file(filename);
    for (float value : data) {
        file << value << " ";
    }
    file.close();
}

// write  a verification function fitting with the datatypes of this program
void verify_results(std::vector<float>& input, std::vector<float>& filter, std::vector<float>& output) {
    int radius = filter.size() / 2;
    int start;
    for (int i = 0; i < input.size(); i++) {
        int temp = 0;
        start = i - radius;
        for (int j = 0; j < filter.size(); j++) {
            if (start + j >= 0 && start + j < input.size()) {
                temp += input[start + j] * filter[j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input file> <mask file> <output file>" << std::endl;
        return 1;
    }
    
    std::vector<float> input, filter, output_gpu;
    readData(argv[1], input);
    readData(argv[2], filter);
    output_gpu.resize(input.size());

    // Allocate memory on GPU
    float *d_input, *d_filter, *d_output;
    cudaMalloc(&d_input, input.size() * sizeof(float));
    cudaMalloc(&d_filter, filter.size() * sizeof(float));
    cudaMalloc(&d_output, input.size() * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter.data(), filter.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // GPU Execution
    int block_size = 256;
    int grid_size = (input.size() + block_size - 1) / block_size;
    
    auto start_gpu = std::chrono::high_resolution_clock::now();
    conv1D_GPU<<<grid_size, block_size>>>(d_input, d_filter, d_output, input.size(), filter.size());
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> gpu_time = end_gpu - start_gpu;
    std::cout << "GPU Execution Time: " << gpu_time.count() << " seconds" << std::endl;

    // Copy result back to CPU
    cudaMemcpy(output_gpu.data(), d_output, input.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);

    // Write output to file
    writeData(argv[3], output_gpu);
    verify_results(input, filter, output_gpu);
    return 0;
}




/*
command to compile and run the code:
nvcc -o k1 k1.cu
k1 i.txt m.txt o.txt
k1 inputfile.txt mask.txt outputfile.txt
naive inputfile.txt mask.txt outputfile2.txt

*/