#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>
#include <math.h>
#include <cmath>
#include <chrono>
#include "common.h"
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#define MAX_MASK_DIM 25

#define CHANNELS 3
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

__constant__ float mask_c[MAX_MASK_DIM * MAX_MASK_DIM];

__global__ void batch_convolution_kernel(unsigned char *red, unsigned char *green, unsigned char *blue, unsigned char *gray,
                                         unsigned int *widths, unsigned int *heights, unsigned int *mem_offset, unsigned int mask_dimension)
{
    // Thread and block indices
    int image_idx = blockIdx.z;                        // Image index in the batch
    int row = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y; // Row index
    int col = blockIdx.x * BLOCK_SIZE_X + threadIdx.x; // Column index
    int channel = threadIdx.z;                         // Channel index (0: red, 1: green, 2: blue)

    // Image dimensions and memory offset
    unsigned int width = widths[image_idx];
    unsigned int height = heights[image_idx];
    int image_offset = mem_offset[image_idx];

    unsigned int mask_radius = mask_dimension / 2;

    // Shared memory for intermediate results
    __shared__ float channel_sums[BLOCK_SIZE_Y][BLOCK_SIZE_X][CHANNELS];

    // Check if the current pixel is within bounds
    if (row < height && col < width)
    {
        float sum = 0.0f;

        // Apply the convolution mask
        for (int mask_row = 0; mask_row < mask_dimension; ++mask_row)
        {
            for (int mask_col = 0; mask_col < mask_dimension; ++mask_col)
            {
                int i_row = row - mask_radius + mask_row;
                int i_col = col - mask_radius + mask_col;

                if (i_row >= 0 && i_row < height && i_col >= 0 && i_col < width)
                {
                    if (channel == 0)
                    { // Red channel
                        sum += red[image_offset + i_row * width + i_col] * mask_c[mask_row * mask_dimension + mask_col];
                    }
                    else if (channel == 1)
                    { // Green channel
                        sum += green[image_offset + i_row * width + i_col] * mask_c[mask_row * mask_dimension + mask_col];
                    }
                    else if (channel == 2)
                    { // Blue channel
                        sum += blue[image_offset + i_row * width + i_col] * mask_c[mask_row * mask_dimension + mask_col];
                    }
                }
            }
        }

        // Clamp the result to [0, 255]
        channel_sums[threadIdx.y][threadIdx.x][channel] = fminf(fmaxf(sum, 0), 255);
        __syncthreads(); // Ensure all threads have written their results

        // Compute grayscale value (only thread with channel = 0 computes it)
        if (channel == 0)
        {
            float r = channel_sums[threadIdx.y][threadIdx.x][0];
            float g = channel_sums[threadIdx.y][threadIdx.x][1];
            float b = channel_sums[threadIdx.y][threadIdx.x][2];
            gray[image_offset + row * width + col] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
        }
    }
}
void rgb2gray_gpu(float *mask, unsigned char *red, unsigned char *green, unsigned char *blue, unsigned char *gray,
                  unsigned int *widths, unsigned int *heights, unsigned int *mem_offset, unsigned int batch_size, unsigned int mask_dimension)
{
    using namespace std::chrono;

    unsigned char *red_d, *green_d, *blue_d, *gray_d;
    unsigned int *widths_d, *heights_d, *mem_offset_d;

    // Allocate memory on the device
    auto start = high_resolution_clock::now();
    cudaMalloc((void **)&red_d, mem_offset[batch_size] * sizeof(unsigned char));
    cudaMalloc((void **)&green_d, mem_offset[batch_size] * sizeof(unsigned char));
    cudaMalloc((void **)&blue_d, mem_offset[batch_size] * sizeof(unsigned char));
    cudaMalloc((void **)&gray_d, mem_offset[batch_size] * sizeof(unsigned char));
    cudaMalloc((void **)&widths_d, batch_size * sizeof(unsigned int));
    cudaMalloc((void **)&heights_d, batch_size * sizeof(unsigned int));
    cudaMalloc((void **)&mem_offset_d, (batch_size + 1) * sizeof(unsigned int));
    cudaDeviceSynchronize();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start).count();
    printf("Device Memory Allocation: %lld ms\n", duration);

    // Copy data to the device
    start = high_resolution_clock::now();
    cudaMemcpy(red_d, red, mem_offset[batch_size] * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(green_d, green, mem_offset[batch_size] * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(blue_d, blue, mem_offset[batch_size] * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(widths_d, widths, batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(heights_d, heights, batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(mem_offset_d, mem_offset, (batch_size + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask_c, mask, mask_dimension * mask_dimension * sizeof(float));
    cudaDeviceSynchronize();
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start).count();
    printf("Data Transfer to Device: %lld ms\n", duration);

    // Determine the maximum image dimensions in the batch
    unsigned int max_width = 0, max_height = 0;
    for (size_t i = 0; i < batch_size; ++i)
    {
        if (widths[i] > max_width)
            max_width = widths[i];
        if (heights[i] > max_height)
            max_height = heights[i];
    }

    // Launch the kernel
    start = high_resolution_clock::now();
    dim3 numThreadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y, CHANNELS);
    dim3 numBlocks((max_width + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
                   (max_height + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
                   batch_size);
    batch_convolution_kernel<<<numBlocks, numThreadsPerBlock>>>(red_d, green_d, blue_d, gray_d, widths_d, heights_d, mem_offset_d, mask_dimension);
    cudaDeviceSynchronize();
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start).count();
    printf("Kernel Execution: %lld ms\n", duration);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    else
    {
        printf("No CUDA Errors\n");
    }

    // Copy results back to the host
    start = high_resolution_clock::now();
    cudaMemcpy(gray, gray_d, mem_offset[batch_size] * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start).count();
    printf("Data Transfer to Host: %lld ms\n", duration);

    // Free device memory
    // start = high_resolution_clock::now();
    cudaFree(red_d);
    cudaFree(green_d);
    cudaFree(blue_d);
    cudaFree(gray_d);
    cudaFree(widths_d);
    cudaFree(heights_d);
    cudaFree(mem_offset_d);
    cudaDeviceSynchronize();
    // stop = high_resolution_clock::now();
    // duration = duration_cast<milliseconds>(stop - start).count();
    printf("Device Memory Deallocation: %lld ms\n", duration);
}

#define STB_IMAGE_IMPLEMENTATION
#include "dependencies/stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "dependencies/stb/stb_image_write.h"

namespace fs = std::filesystem;

// Count the number of images in the input folder
int count_images(const std::string &input_folder)
{
    int count = 0;
    for (const auto &entry : fs::directory_iterator(input_folder))
    {
        if (entry.is_regular_file())
        {
            count++;
        }
    }
    return count;
}

// Load a batch of images from the input folder and store their filenames
std::vector<std::pair<std::string, unsigned char *>> load_batch(const std::string &input_folder, int batch_size, int start_index, int total_images, int *width, int *height)
{
    std::vector<std::pair<std::string, unsigned char *>> batch;
    int count = 0;

    for (const auto &entry : fs::directory_iterator(input_folder))
    {
        if (count >= start_index && count < start_index + batch_size && count < total_images && entry.is_regular_file())
        {
            std::string file_path = entry.path().string();
            std::string filename = entry.path().filename().string(); // Extract the filename

            // Load image using STB
            int w, h, channels;
            unsigned char *image_data = stbi_load(file_path.c_str(), &w, &h, &channels, 0);
            if (image_data)
            {
                if (channels == 3 || channels == 4)
                { // Accept both RGB and RGBA images
                    unsigned char *rgb_data = image_data;
                    if (channels == 4)
                    {
                        // Convert RGBA to RGB by stripping the alpha channel
                        rgb_data = (unsigned char *)malloc(w * h * 3 * sizeof(unsigned char));
                        if (!rgb_data)
                        {
                            fprintf(stderr, "Error: Memory allocation failed for RGB conversion\n");
                            stbi_image_free(image_data);
                            continue;
                        }
                        for (int i = 0; i < w * h; ++i)
                        {
                            rgb_data[i * 3] = image_data[i * 4];         // Red channel
                            rgb_data[i * 3 + 1] = image_data[i * 4 + 1]; // Green channel
                            rgb_data[i * 3 + 2] = image_data[i * 4 + 2]; // Blue channel
                        }
                        stbi_image_free(image_data); // Free the original RGBA data
                    }
                    batch.emplace_back(filename, rgb_data);
                    width[count - start_index] = w;
                    height[count - start_index] = h;
                }
                else
                {
                    fprintf(stderr, "Skipping image %s: Expected 3 or 4 channels (RGB/RGBA), got %d channels\n", file_path.c_str(), channels);
                    stbi_image_free(image_data); // Free the image data if it doesn't match
                }
            }
            else
            {
                fprintf(stderr, "Failed to load image: %s\n", file_path.c_str());
            }
        }
        count++;
        if (batch.size() >= batch_size || count >= total_images)
        {
            break;
        }
    }

    return batch;
}

// Function to save a batch of grayscale images
void save_batch(const unsigned char *gray, unsigned int *widths, unsigned int *heights, unsigned int *mem_offset, const std::string &output_folder, const std::vector<std::string> &filenames, const char *prefix_id)
{
    for (size_t i = 0; i < filenames.size(); ++i)
    {
        std::string output_path = output_folder + "/" + prefix_id + filenames[i];
        stbi_write_png(output_path.c_str(), widths[i], heights[i], 1, gray + mem_offset[i], widths[i]);
    }
}

int main(int argc, char *argv[])
{
    printf("----- Kernel 1 (Different Size Images) -----\n");
    // Process command-line arguments
    if (argc < 5)
    {
        fprintf(stderr, "Usage: %s <input_folder> <output_folder> <batch_size> <mask_file_path>\n", argv[0]);
        return 1;
    }

    std::string input_folder = argv[1];
    std::string output_folder = argv[2];
    unsigned int batch_size = std::stoi(argv[3]);
    const char *mask_file = argv[4];

    // Count the total number of images in the input folder
    int total_images = count_images(input_folder);
    if (total_images == 0)
    {
        fprintf(stderr, "No images found in the input folder.\n");
        return 1;
    }

    printf("Found %d images in the input folder.\n", total_images);
    printf("--------------------------------------------------\n");

    // Calculate the number of batches
    int num_batches = std::ceil((float)total_images / batch_size);

    int mask_dimension;
    float **mask = read_mask(mask_file, &mask_dimension);
    float *flattened_mask = flatten_mask(mask, mask_dimension);

    const char *prefix_id = ""; // Prefix for the output filenames

    // Batches are processed sequentially
    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx)
    {
        printf("*********** Processing batch %d ************\n", batch_idx + 1);

        int start_index = batch_idx * batch_size;

        int batch_size_actual = (start_index + batch_size > total_images) ? (total_images - start_index) : batch_size;

        // Load the current batch of images and filenames
        int *widths = (int *)malloc(batch_size_actual * sizeof(int));
        int *heights = (int *)malloc(batch_size_actual * sizeof(int));
        std::vector<std::pair<std::string, unsigned char *>> batch = load_batch(input_folder, batch_size_actual, start_index, total_images, widths, heights);
        if (batch.empty())
        {
            fprintf(stderr, "Failed to load batch %d\n", batch_idx);
            continue;
        }

        // Calculate memory offsets
        unsigned int *mem_offset = (unsigned int *)malloc((batch_size_actual + 1) * sizeof(unsigned int));
        mem_offset[0] = 0;
        for (size_t i = 1; i <= batch_size_actual; ++i)
        {
            mem_offset[i] = mem_offset[i - 1] + widths[i - 1] * heights[i - 1];
        }

        // Collect filenames
        std::vector<std::string> filenames;
        for (const auto &pair : batch)
        {
            filenames.push_back(pair.first); // Save the filenames
        }

        // Allocate memory for the batch of grayscale images
        unsigned char *gray = (unsigned char *)malloc(mem_offset[batch_size_actual] * sizeof(unsigned char));
        if (!gray)
        {
            fprintf(stderr, "Error: Memory allocation failed for grayscale images\n");
            return 1;
        }

        // Allocate memory for the batch of RGB images
        unsigned char *red = (unsigned char *)malloc(mem_offset[batch_size_actual] * sizeof(unsigned char));
        unsigned char *green = (unsigned char *)malloc(mem_offset[batch_size_actual] * sizeof(unsigned char));
        unsigned char *blue = (unsigned char *)malloc(mem_offset[batch_size_actual] * sizeof(unsigned char));
        if (!red || !green || !blue)
        {
            fprintf(stderr, "Error: Memory allocation failed for RGB images\n");
            free(gray);
            return 1;
        }

        // Copy RGB data from the batch to the flat arrays
        for (size_t i = 0; i < batch.size(); ++i)
        {
            unsigned char *image_data = batch[i].second;
            for (int j = 0; j < widths[i] * heights[i]; ++j)
            {
                red[mem_offset[i] + j] = image_data[j * 3];       // Red channel
                green[mem_offset[i] + j] = image_data[j * 3 + 1]; // Green channel
                blue[mem_offset[i] + j] = image_data[j * 3 + 2];  // Blue channel
            }
        }

        // Call the GPU function (process the entire batch in one call)
        rgb2gray_gpu(flattened_mask, red, green, blue, gray, (unsigned int *)widths, (unsigned int *)heights, mem_offset, batch_size_actual, mask_dimension);

        // Save the batch of grayscale images with original filenames
        save_batch(gray, (unsigned int *)widths, (unsigned int *)heights, mem_offset, output_folder, filenames, prefix_id);

        // Free memory
        free(red);
        free(green);
        free(blue);
        free(gray);
        for (const auto &pair : batch)
        {
            if (pair.second)
            {
                free(pair.second); // Free the RGB data (may have been allocated for RGBA conversion)
            }
        }

        printf("Batch %d processed and saved.\n", batch_idx + 1);
        printf("--------------------------------------------------\n");
    }

    // Free the mask memory
    for (int i = 0; i < mask_dimension; ++i)
    {
        free(mask[i]);
    }
    free(mask);
    free(flattened_mask);

    printf("Grayscale conversion completed. Results saved to %s.\n", output_folder.c_str());
    return 0;
}

/*
cd E:\Spring25\Selected_topics\Lab5
nvcc -o k1 kernel1_m.cu -std=c++17
kernel2 input_images output_images/kernel2_cu 2 masks/mask.txt
*/