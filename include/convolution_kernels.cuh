#pragma once
#include <opencv2/opencv.hpp>

void apply_custom_convolution(const cv::Mat &input, cv::Mat &output, const std::string &filter_type);