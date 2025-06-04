#include <iostream>
#include <vector>
#include <fstream>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <iomanip>

#define TILE_SIZE 4  // Each thread computes TILE_SIZE elements

// GPU Kernel for 1D Convolution with Output Tiling
__global__ void conv1D_GPU(const float* __restrict__ input, const float* __restrict__ filter, float* output, int input_size, int filter_size) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int radius = filter_size / 2;

    for (int t = 0; t < TILE_SIZE; t++) {
        int idx = tid * TILE_SIZE + t;
        if (idx < input_size) {
            float sum = 0.0f;
            for (int j = 0; j < filter_size; ++j) {
                int input_idx = idx + j - radius;
                if (input_idx >= 0 && input_idx < input_size) {
                    sum += input[input_idx] * filter[j];
                }
            }
            output[idx] = sum;
        }
    }
}

// Function to read data from file
void readData(const std::string& filename, std::vector<float>& data) {
    std::ifstream file(filename);
    int size;
    file >> size;
    data.resize(size);
    for (int i = 0; i < size; ++i) {
        file >> data[i];
    }
}

// // Function to write data to file
// void writeData(const std::string& filename, const std::vector<float>& data) {
//     std::ofstream file(filename);
//     // for (float value : data) {
//     //     file << value << " ";
//     // }
//     // write the data such that they're printed in normal notation not e+
//     for (int i = 0; i < data.size(); i++) {
//         file << data[i] << " ";
//     }
// }
void writeData(const std::string& filename, const std::vector<float>& data) {
    std::ofstream file(filename);
    // Set the output format to fixed-point with a specified precision
    file << std::fixed << std::setprecision(6);  // Adjust precision as needed
    for (int i = 0; i < data.size(); i++) {
        file << data[i] << " ";
    }
    file << std::endl;  // Optionally add a newline after data
}
// Verification function
void verify_results(const std::vector<float>& input, const std::vector<float>& filter, const std::vector<float>& output) {
    int radius = filter.size() / 2;
    for (size_t i = 0; i < input.size(); i++) {
        float temp = 0;
        for (size_t j = 0; j < filter.size(); j++) {
            int idx = i + j - radius;
            if (idx >= 0 && idx < input.size()) {
                temp += input[idx] * filter[j];
            }
        }
        if (abs(temp - output[i]) > 1e-5) return;
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
    
    // Start profiling
    cudaProfilerStart();

    // GPU Execution
    int block_size = 256;
    int grid_size = (input.size() + TILE_SIZE * block_size - 1) / (TILE_SIZE * block_size);
    conv1D_GPU<<<grid_size, block_size>>>(d_input, d_filter, d_output, input.size(), filter.size());
    cudaDeviceSynchronize();

    // Stop profiling
    cudaProfilerStop();

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


// #include <iostream>
// #include <vector>
// #include <fstream>
// #include <cuda.h>

// #define TILE_SIZE 4  // Each thread computes TILE_SIZE elements

// // GPU Kernel for 1D Convolution with Output Tiling
// __global__ void conv1D_GPU(const float* input, const float* filter, float* output, int input_size, int filter_size) {
//     int idx = (blockIdx.x * blockDim.x + threadIdx.x) * TILE_SIZE;
//     int radius = filter_size / 2;

//     for (int t = 0; t < TILE_SIZE; t++) {
//         int tid = idx + t;
//         if (tid < input_size) {
//             float sum = 0.0f;
//             for (int j = 0; j < filter_size; ++j) {
//                 int input_idx = tid + j - radius;
//                 if (input_idx >= 0 && input_idx < input_size) {
//                     sum += input[input_idx] * filter[j];
//                 }
//             }
//             output[tid] = sum;
//         }
//     }
// }

// // Function to read data from file
// void readData(const std::string& filename, std::vector<float>& data) {
//     std::ifstream file(filename);
//     int size;
//     file >> size;
//     data.resize(size);
//     for (int i = 0; i < size; ++i) {
//         file >> data[i];
//     }
//     file.close();
// }

// // Function to write data to file
// void writeData(const std::string& filename, const std::vector<float>& data) {
//     std::ofstream file(filename);
//     for (float value : data) {
//         file << value << " ";
//     }
//     file.close();
// }

// // Verification function
// void verify_results(std::vector<float>& input, std::vector<float>& filter, std::vector<float>& output) {
//     int radius = filter.size() / 2;
//     for (int i = 0; i < input.size(); i++) {
//         float temp = 0;
//         for (int j = 0; j < filter.size(); j++) {
//             int idx = i + j - radius;
//             if (idx >= 0 && idx < input.size()) {
//                 temp += input[idx] * filter[j];
//             }
//         }
//         if (abs(temp - output[i]) > 1e-5) {
//             // std::cerr << "mismatch at index " << i << " expected " << temp << " got " << output[i] << std::endl;
//             return;
//         }
//     }
//     // std::cout << "Verification successful" << std::endl;
// }

// int main(int argc, char* argv[]) {
//     if (argc != 4) {
//         std::cerr << "Usage: " << argv[0] << " <input file> <mask file> <output file>" << std::endl;
//         return 1;
//     }
    
//     std::vector<float> input, filter, output_gpu;
//     readData(argv[1], input);
//     readData(argv[2], filter);
//     output_gpu.resize(input.size());

//     // Allocate memory on GPU
//     float *d_input, *d_filter, *d_output;
//     cudaMalloc(&d_input, input.size() * sizeof(float));
//     cudaMalloc(&d_filter, filter.size() * sizeof(float));
//     cudaMalloc(&d_output, input.size() * sizeof(float));

//     // Copy data to GPU
//     cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_filter, filter.data(), filter.size() * sizeof(float), cudaMemcpyHostToDevice);
    
//     // GPU Execution
//     int block_size = 256;
//     int grid_size = (input.size() + TILE_SIZE * block_size - 1) / (TILE_SIZE * block_size);
//     conv1D_GPU<<<grid_size, block_size>>>(d_input, d_filter, d_output, input.size(), filter.size());
//     cudaDeviceSynchronize();

//     // Copy result back to CPU
//     cudaMemcpy(output_gpu.data(), d_output, input.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
//     // Free GPU memory
//     cudaFree(d_input);
//     cudaFree(d_filter);
//     cudaFree(d_output);

//     // Write output to file
//     writeData(argv[3], output_gpu);
//     verify_results(input, filter, output_gpu);
//     return 0;
// }
