#pragma once

__global__ void convolution_2d_kernel(const unsigned char *input, unsigned char *output, int width, int height, int input_pitch, int output_pitch);