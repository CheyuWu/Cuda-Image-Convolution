#include "convolution_kernels.cuh"
void apply_custom_convolution(const cv::Mat &input, cv::Mat &output, const std::string &filter_type)
{
    // TODO: implement CUDA kernel here
    output = input.clone();

    
}
