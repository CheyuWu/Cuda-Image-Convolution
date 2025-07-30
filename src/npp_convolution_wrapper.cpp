#include "npp_convolution_wrapper.hpp"
#include <nppi.h>
#include <npp.h>
#include <nppdefs.h>
#include <cuda_runtime.h>
#include <iostream>

void apply_npp_convolution(const cv::Mat &input, cv::Mat &output, const std::string &filter_type)
{
    // TODO: implement NPP-based convolution
    std::cerr << "[NPP] Convolution not implemented yet.\n";
    output = input.clone();
}