#include "npp_convolution_wrapper.hpp"
#include <nppi.h>
#include <npp.h>
#include <nppdefs.h>
#include <cuda_runtime.h>
#include <iostream>

void apply_npp_convolution(const cv::Mat &input, cv::Mat &output, const std::string &filter_type)
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
    
    if (input.channels() != 1 || input.depth() != CV_8U)
    {
        std::cerr << "[NPP] Only grayscale CV_8UC1 images are supported.\n";
        output = input.clone();
        return;
    }

    NppiSize oSizeROI = {input.cols, input.rows};
    NppiSize oSrcSize = oSizeROI;
    NppiPoint oSrcOffset = {0, 0};

    int srcStep = input.step;
    int dstStep = input.cols;

    Npp8u *pSrc, *pDst;
    cudaMalloc((void **)&pSrc, input.cols * input.rows);
    cudaMalloc((void **)&pDst, input.cols * input.rows);
    cudaMemcpy(pSrc, input.data, input.cols * input.rows, cudaMemcpyHostToDevice);

    NppStatus status;

    if (filter_type == "sobel")
    {
        status = nppiFilterSobelHorizBorder_8u_C1R(
            pSrc, srcStep, oSrcSize, oSrcOffset,
            pDst, dstStep, oSizeROI,
            NPP_BORDER_REPLICATE);
    }
    else if (filter_type == "gaussian")
    {
        status = nppiFilterGaussBorder_8u_C1R(
            pSrc, srcStep, oSrcSize, oSrcOffset,
            pDst, dstStep, oSizeROI,
            NPP_MASK_SIZE_3_X_3,
            NPP_BORDER_REPLICATE);
    }
    else
    {
        std::cerr << "[NPP] Unknown filter type: " << filter_type << "\n";
        cudaFree(pSrc);
        cudaFree(pDst);
        output = input.clone();
        return;
    }

    if (status != NPP_SUCCESS)
    {
        std::cerr << "[NPP] Error: NPP filter failed with code " << status << "\n";
        cudaFree(pSrc);
        cudaFree(pDst);
        output = input.clone();
        return;
    }

    output.create(input.rows, input.cols, CV_8UC1);
    cudaMemcpy(output.data, pDst, input.cols * input.rows, cudaMemcpyDeviceToHost);
    cudaFree(pSrc);
    cudaFree(pDst);
}
