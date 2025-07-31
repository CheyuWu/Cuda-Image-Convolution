#include "convolution_kernels.cuh"
#include <cuda_runtime.h>
#include <iostream>

#define KERNEL_SIZE 3
#define TILE_WIDTH 16

__constant__ float d_kernel[9];

__global__ void convolution_2d_kernel(const unsigned char *input, unsigned char *output, int width, int height, int input_pitch, int output_pitch)
{
    __shared__ unsigned char tile[TILE_WIDTH + KERNEL_SIZE - 1][TILE_WIDTH + KERNEL_SIZE - 1];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * TILE_WIDTH + ty;
    int col_o = blockIdx.x * TILE_WIDTH + tx;
    int row_i = row_o - KERNEL_SIZE / 2;
    int col_i = col_o - KERNEL_SIZE / 2;

    if (row_i >= 0 && row_i < height && col_i >= 0 && col_i < width)
    {
        tile[ty][tx] = input[row_i * input_pitch + col_i];
    }
    else
    {
        tile[ty][tx] = 0;
    }
    __syncthreads();

    if (tx < TILE_WIDTH && ty < TILE_WIDTH && row_o < height && col_o < width)
    {
        float acc = 0.0f;
        for (int i = 0; i < KERNEL_SIZE; ++i)
        {
            for (int j = 0; j < KERNEL_SIZE; ++j)
            {
                acc += d_kernel[i * KERNEL_SIZE + j] * tile[ty + i][tx + j];
            }
        }
        acc = fminf(fmaxf(acc, 0.0f), 255.0f);
        output[row_o * output_pitch + col_o] = static_cast<unsigned char>(acc);
    }
}

void apply_custom_convolution(const cv::Mat &input, cv::Mat &output, const std::string &filter_type)
{
    cv::Mat gray;
    if (input.channels() == 3)
    {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        gray = input;
    }

    float h_kernel[9];
    if (filter_type == "sobel")
    {
        float sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
        std::copy(sobel_x, sobel_x + 9, h_kernel);
    }
    else if (filter_type == "gaussian")
    {
        float gaussian[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
        for (int i = 0; i < 9; ++i)
            h_kernel[i] = gaussian[i] / 16.0f;
    }
    else if (filter_type == "sharpen")
    {
        float sharpen[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0};
        std::copy(sharpen, sharpen + 9, h_kernel);
    }
    else
    {
        std::cerr << "[Custom] Unknown filter type: " << filter_type << "\n";
        output = input.clone();
        return;
    }

    cudaMemcpyToSymbol(d_kernel, h_kernel, sizeof(float) * 9);

    int width = gray.cols;
    int height = gray.rows;
    size_t input_pitch, output_pitch;
    unsigned char *d_input, *d_output;

    cudaMallocPitch(&d_input, &input_pitch, width * sizeof(unsigned char), height);
    cudaMallocPitch(&d_output, &output_pitch, width * sizeof(unsigned char), height);
    cudaMemcpy2D(d_input, input_pitch, gray.data, width, width, height, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH + KERNEL_SIZE - 1, TILE_WIDTH + KERNEL_SIZE - 1);
    dim3 dimGrid((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_WIDTH - 1) / TILE_WIDTH);
    convolution_2d_kernel<<<dimGrid, dimBlock>>>(d_input, d_output, width, height, input_pitch, output_pitch);
    cudaDeviceSynchronize();

    output.create(height, width, CV_8UC1);
    cudaMemcpy2D(output.data, width, d_output, output_pitch, width, height, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
