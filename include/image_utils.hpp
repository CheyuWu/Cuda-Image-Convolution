#include <string>
#include <opencv2/opencv.hpp>

cv::Mat load_grayscale_image(const std::string &path);

void save_image(const std::string &path, const cv::Mat &image);