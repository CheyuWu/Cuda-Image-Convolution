#include <iostream>
#include <string>
#include "convolution_kernels.hpp"
#include "npp_convolution_wrapper.hpp"
#include "image_utils.hpp"

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cerr << "Usage: ./main <input_path> <output_path> <filter_type> [custom|npp]\n";
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    std::string filter_type = argv[3];
    std::string mode = argc > 4 ? argv[4] : "custom";

    cv::Mat input = load_grayscale_image(input_path);
    cv::Mat output;

    if (mode == "custom")
    {
        apply_custom_convolution(input, output, filter_type);
    }
    else if (mode == "npp")
    {
        apply_npp_convolution(input, output, filter_type);
    }
    else
    {
        std::cerr << "Unknown mode: " << mode << "\n";
        return 1;
    }

    save_image(output_path, output);
    std::cout << "Output saved to " << output_path << "\n";
    return 0;
}