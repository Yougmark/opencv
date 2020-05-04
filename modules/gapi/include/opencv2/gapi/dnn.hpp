// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#ifndef OPENCV_GAPI_DNN_HPP
#define OPENCV_GAPI_DNN_HPP

#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include <utility> // std::tuple

#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/gmat.hpp>
#include <opencv2/gapi/gscalar.hpp>


/** \defgroup gapi_dnn G-API Image processing functionality
@{
@}
 */

namespace cv { namespace gapi {

namespace dnn {
    using GMat2 = std::tuple<GMat,GMat>;
    using GMat3 = std::tuple<GMat,GMat,GMat>; // FIXME: how to avoid this?

    G_TYPED_KERNEL(GConvolve, <GMat(GMat,cv::dnn::Net*,std::vector<cv::Mat>*, std::vector<std::string>*)>,         "org.opencv.imgproc.filters.convolve"){
        static GMatDesc outMeta(GMatDesc in, cv::dnn::Net*, std::vector<cv::Mat>*, std::vector<std::string>*) {
            return in;
        }
    };

} //namespace dnn

GAPI_EXPORTS GMat convolve(const GMat& src, cv::dnn::Net* net, std::vector<cv::Mat>* outs, std::vector<std::string>* outNames);

} //namespace gapi
} //namespace cv

#endif // OPENCV_GAPI_DNN_HPP
