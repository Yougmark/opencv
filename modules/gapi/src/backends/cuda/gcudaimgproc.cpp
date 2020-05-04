// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"

#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/cuda/imgproc.hpp>
#include "backends/cuda/gcudaimgproc.hpp"


GAPI_CUDA_KERNEL(GCUDAConvolve, cv::gapi::imgproc::GConvolve)
{
    static void run(const cv::Mat& in, cv::dnn::Net* net, std::vector<cv::Mat>* outs, std::vector<std::string>* outNames, cv::Mat &out)
    {
        printf("before forward in cuda backend\n");
        net->forward(*outs, *outNames);
        printf("after forward in cuda backend\n");
    }
};


cv::gapi::GKernelPackage cv::gapi::imgproc::cuda::kernels()
{
    static auto pkg = cv::gapi::kernels
        < GCUDAConvolve
        >();
    return pkg;
}
