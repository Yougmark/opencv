// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#include "precomp.hpp"

#include <opencv2/gapi/gscalar.hpp>
#include <opencv2/gapi/gcall.hpp>
#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/dnn.hpp>

namespace cv { namespace gapi {

GMat convolve(const GMat& src, cv::dnn::Net* net, std::vector<cv::Mat>*
               outs, std::vector<std::string>* outNames)
{
    return imgproc::GConvolve::on(src, net, outs, outNames);
}

} //namespace gapi
} //namespace cv
