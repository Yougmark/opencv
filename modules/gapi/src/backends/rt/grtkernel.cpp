// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation

#include "precomp.hpp"

#include <cassert>

#include <opencv2/gapi/rt/grtkernel.hpp>

const cv::Mat &cv::GRTContext::inMat(int input)
{
    return inArg<cv::Mat>(input);
}

cv::Mat &cv::GRTContext::outMatR(int output)
{
    return *util::get<cv::Mat *>(m_results.at(output));
}

const cv::Scalar &cv::GRTContext::inVal(int input)
{
    return inArg<cv::Scalar>(input);
}

cv::Scalar &cv::GRTContext::outValR(int output)
{
    return *util::get<cv::Scalar *>(m_results.at(output));
}

cv::detail::VectorRef &cv::GRTContext::outVecRef(int output)
{
    return util::get<cv::detail::VectorRef>(m_results.at(output));
}

cv::detail::OpaqueRef &cv::GRTContext::outOpaqueRef(int output)
{
    return util::get<cv::detail::OpaqueRef>(m_results.at(output));
}

cv::GRTKernel::GRTKernel() {}

cv::GRTKernel::GRTKernel(const GRTKernel::F &f) : m_f(f) {}

void cv::GRTKernel::apply(GRTContext &ctx)
{
    GAPI_Assert(m_f);
    m_f(ctx);
}
