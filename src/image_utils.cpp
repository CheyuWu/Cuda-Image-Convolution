#include <opencv2/opencv.hpp>

cv::Mat load_grayscale_image(const std::string &path)
{
    return cv::imread(path, cv::IMREAD_GRAYSCALE);
}

void save_image(const std::string &path, const cv::Mat &image)
{
    cv::imwrite(path, image);
}
