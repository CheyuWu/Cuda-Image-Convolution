#pragma once
#include <opencv2/opencv.hpp>


__global__ void convolution_2d_kernel(const unsigned char *input, unsigned char *output, int width, int height, int input_pitch, int output_pitch);
void apply_custom_convolution(const cv::Mat &input, cv::Mat &output, const std::string &filter_type);