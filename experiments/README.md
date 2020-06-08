
## Compile DNN with CUDA backend
```
mkdir build
cd build
# OPENCL for DNN doesn't work on NVIDIA GPU but only Intel GPU, so we disable it.
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_CUDA=ON -D WITH_CUDNN=ON -D BUILD_EXAMPLES=Yes -D ENABLE_CXX11=Yes -D CUDA_ARCH=61 -D CUDA_ARCH_BIN=61 -D OPENCV_DNN_CUDA=ON -D OPENCV_DNN_OPENCL=OFF -D OPENCV_EXTRA_MODULES_PATH=/home/yang/Projects/opencv_contrib/modules/ -D OPENCV_TEST_DATA_PATH=/home/yang/Projects/opencv_extra/test_data ..
make opencv_test_dnn opencv_perf_dnn -j 15
```

cmake output:
 
```
yang@yang-desktop:~/Projects/opencv/build$ cmake -D CMAKE_BUILD_TYPE=Release -D
CMAKE_INSTALL_PREFIX=/usr/local -D WITH_CUDA=ON -D WITH_CUDNN=ON -D
BUILD_EXAMPLES=Yes -D ENABLE_CXX11=Yes -D CUDA_ARCH=61 -D CUDA_ARCH_BIN=61 -D
OPENCV_DNN_CUDA=ON -D OPENCV_DNN_OPENCL=OFF -D
OPENCV_EXTRA_MODULES_PATH=/home/yang/Projects/opencv_contrib/modules/ -D
OPENCV_TEST_DATA_PATH=/home/yang/Projects/opencv_extra/test_data ..
-- Detected processor: x86_64
-- Looking for ccache - not found
-- Found ZLIB: /usr/lib/x86_64-linux-gnu/libz.so (found suitable version
"1.2.11", minimum required is "1.2.3")
-- Could NOT find Jasper (missing: JASPER_LIBRARIES JASPER_INCLUDE_DIR)
-- Found ZLIB: /usr/lib/x86_64-linux-gnu/libz.so (found version "1.2.11")
-- Checking for module 'gtk+-3.0'
--   No package 'gtk+-3.0' found
-- found Intel IPP (ICV version): 2019.0.0 [2019.0.0 Gold]
-- at: /home/yang/Projects/opencv/build/3rdparty/ippicv/ippicv_lnx/icv
-- found Intel IPP Integration Wrappers sources: 2019.0.0
-- at: /home/yang/Projects/opencv/build/3rdparty/ippicv/ippicv_lnx/iw
-- CUDA detected: 10.2
-- CUDA NVCC target flags: -gencode;arch=compute_61,code=sm_61;-D_FORCE_INLINES
-- Could not find OpenBLAS include. Turning OpenBLAS_FOUND off
-- Could not find OpenBLAS lib. Turning OpenBLAS_FOUND off
-- Could NOT find Atlas (missing: Atlas_CBLAS_INCLUDE_DIR
Atlas_CLAPACK_INCLUDE_DIR Atlas_CBLAS_LIBRARY Atlas_BLAS_LIBRARY
Atlas_LAPACK_LIBRARY)
-- A library with BLAS API not found. Please specify library location.
-- LAPACK requires BLAS
-- A library with LAPACK API not found. Please specify library location.
-- Could NOT find JNI (missing: JAVA_INCLUDE_PATH JAVA_INCLUDE_PATH2
JAVA_AWT_INCLUDE_PATH)
-- Could NOT find Pylint (missing: PYLINT_EXECUTABLE)
-- Could NOT find Flake8 (missing: FLAKE8_EXECUTABLE)
-- VTK is not found. Please set -DVTK_DIR in CMake to VTK build directory, or to
VTK install subdirectory with VTKConfig.cmake file
-- OpenCV Python: during development append to PYTHONPATH:
/home/yang/Projects/opencv/build/python_loader
-- Checking for module 'gstreamer-base-1.0'
--   No package 'gstreamer-base-1.0' found
-- Checking for module 'gstreamer-app-1.0'
--   No package 'gstreamer-app-1.0' found
-- Checking for module 'gstreamer-riff-1.0'
--   No package 'gstreamer-riff-1.0' found
-- Checking for module 'gstreamer-pbutils-1.0'
--   No package 'gstreamer-pbutils-1.0' found
-- Caffe:   NO
-- Protobuf:   NO
-- Glog:   YES
-- freetype2:   YES (ver 21.0.15)
-- harfbuzz:    YES (ver 1.7.2)
-- Could NOT find HDF5 (missing: HDF5_LIBRARIES HDF5_INCLUDE_DIRS) (found
version "")
-- Module opencv_ovis disabled because OGRE3D was not found
-- No preference for use of exported gflags CMake configuration set, and no
hints for include/library directories provided. Defaulting to preferring an
installed/exported gflags CMake configuration if available.
-- Found installed version of gflags: /usr/lib/x86_64-linux-gnu/cmake/gflags
-- Detected gflags version: 2.2.1
-- Checking SFM deps... TRUE
-- Module opencv_sfm disabled because the following dependencies are not found:
Eigen
-- Checking for module 'tesseract'
--   No package 'tesseract' found
-- Tesseract:   NO
-- Registering hook 'INIT_MODULE_SOURCES_opencv_dnn':
/home/yang/Projects/opencv/modules/dnn/cmake/hooks/INIT_MODULE_SOURCES_opencv_dnn.cmake
-- opencv_dnn: filter out ocl4dnn source code
-- OpenCL samples are skipped: OpenCL SDK is required
--
-- General configuration for OpenCV 4.2.0-dev
=====================================
--   Version control:               4.2.0-34-g43a91f82fe
--
--   Extra modules:
--     Location (extra):            /home/yang/Projects/opencv_contrib/modules
--     Version control (extra):     4.2.0-4-g648db235
--
--   Platform:
--     Timestamp:                   2020-01-05T02:45:32Z
--     Host:                        Linux 5.0.0-37-generic x86_64
--     CMake:                       3.10.2
--     CMake generator:             Unix Makefiles
--     CMake build tool:            /usr/bin/make
--     Configuration:               Release
--
--   CPU/HW features:
--     Baseline:                    SSE SSE2 SSE3
--       requested:                 SSE3
--     Dispatched code generation:  SSE4_1 SSE4_2 FP16 AVX AVX2 AVX512_SKX
--       requested:                 SSE4_1 SSE4_2 AVX FP16 AVX2 AVX512_SKX
--       SSE4_1 (16 files):         + SSSE3 SSE4_1
--       SSE4_2 (2 files):          + SSSE3 SSE4_1 POPCNT SSE4_2
--       FP16 (1 files):            + SSSE3 SSE4_1 POPCNT SSE4_2 FP16 AVX
--       AVX (5 files):             + SSSE3 SSE4_1 POPCNT SSE4_2 AVX
--       AVX2 (29 files):           + SSSE3 SSE4_1 POPCNT SSE4_2 FP16 FMA3 AVX
AVX2
--       AVX512_SKX (6 files):      + SSSE3 SSE4_1 POPCNT SSE4_2 FP16 FMA3 AVX
AVX2 AVX_512F AVX512_COMMON AVX512_SKX
--
--   C/C++:
--     Built as dynamic libs?:      YES
--     C++ Compiler:                /usr/bin/c++  (ver 7.4.0)
--     C++ flags (Release):         -fsigned-char -W -Wall -Werror=return-type
-Werror=non-virtual-dtor -Werror=address -Werror=sequence-point -Wformat
-Werror=format-security -Wmissing-declarations -Wundef -Winit-self
-Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Winit-self
-Wsuggest-override -Wno-delete-non-virtual-dtor -Wno-comment
-Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option
-Wno-long-long -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections
-msse -msse2 -msse3 -fvisibility=hidden -fvisibility-inlines-hidden -O3 -DNDEBUG
-DNDEBUG
--     C++ flags (Debug):           -fsigned-char -W -Wall -Werror=return-type
-Werror=non-virtual-dtor -Werror=address -Werror=sequence-point -Wformat
-Werror=format-security -Wmissing-declarations -Wundef -Winit-self
-Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Winit-self
-Wsuggest-override -Wno-delete-non-virtual-dtor -Wno-comment
-Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option
-Wno-long-long -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections
-msse -msse2 -msse3 -fvisibility=hidden -fvisibility-inlines-hidden -g  -O0
-DDEBUG -D_DEBUG
--     C Compiler:                  /usr/bin/cc
--     C flags (Release):           -fsigned-char -W -Wall -Werror=return-type
-Werror=non-virtual-dtor -Werror=address -Werror=sequence-point -Wformat
-Werror=format-security -Wmissing-declarations -Wmissing-prototypes
-Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -Wuninitialized
-Winit-self -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow
-fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer
-ffunction-sections -fdata-sections  -msse -msse2 -msse3 -fvisibility=hidden -O3
-DNDEBUG  -DNDEBUG
--     C flags (Debug):             -fsigned-char -W -Wall -Werror=return-type
-Werror=non-virtual-dtor -Werror=address -Werror=sequence-point -Wformat
-Werror=format-security -Wmissing-declarations -Wmissing-prototypes
-Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -Wuninitialized
-Winit-self -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow
-fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer
-ffunction-sections -fdata-sections  -msse -msse2 -msse3 -fvisibility=hidden -g
-O0 -DDEBUG -D_DEBUG
--     Linker flags (Release):      -Wl,--gc-sections
--     Linker flags (Debug):        -Wl,--gc-sections
--     ccache:                      NO
--     Precompiled headers:         NO
--     Extra dependencies:          m pthread cudart_static -lpthread dl rt nppc
nppial nppicc nppicom nppidei nppif nppig nppim nppist nppisu nppitc npps cublas
cudnn cufft -L/usr/local/cuda-10.2/lib64 -L/usr/lib/x86_64-linux-gnu
--     3rdparty dependencies:
--
--   OpenCV modules:
--     To be built:                 aruco bgsegm bioinspired calib3d ccalib core
cudaarithm cudabgsegm cudacodec cudafeatures2d cudafilters cudaimgproc
cudalegacy cudaobjdetect cudaoptflow cudastereo cudawarping cudev datasets dnn
dnn_objdetect dnn_superres dpm face features2d flann freetype fuzzy gapi hfs
highgui img_hash imgcodecs imgproc line_descriptor ml objdetect optflow
phase_unwrapping photo plot python2 python3 quality rapid reg rgbd saliency
shape stereo stitching structured_light superres surface_matching text tracking
ts video videoio videostab xfeatures2d ximgproc xobjdetect xphoto
--     Disabled:                    world
--     Disabled by dependency:      -
--     Unavailable:                 cnn_3dobj cvv hdf java js matlab ovis sfm
viz
--     Applications:                tests perf_tests examples apps
--     Documentation:               NO
--     Non-free algorithms:         NO
--
--   GUI:
--     GTK+:                        YES (ver 2.24.32)
--       GThread :                  YES (ver 2.56.4)
--       GtkGlExt:                  NO
--     VTK support:                 NO
--
--   Media I/O:
--     ZLib:                        /usr/lib/x86_64-linux-gnu/libz.so (ver
1.2.11)
--     JPEG:                        /usr/lib/x86_64-linux-gnu/libjpeg.so (ver
80)
--     WEBP:                        build (ver encoder: 0x020e)
--     PNG:                         /usr/lib/x86_64-linux-gnu/libpng.so (ver
1.6.34)
--     TIFF:                        /usr/lib/x86_64-linux-gnu/libtiff.so (ver 42
/ 4.0.9)
--     JPEG 2000:                   build (ver 1.900.1)
--     OpenEXR:                     build (ver 2.3.0)
--     HDR:                         YES
--     SUNRASTER:                   YES
--     PXM:                         YES
--     PFM:                         YES
--
--   Video I/O:
--     DC1394:                      YES (2.2.5)
--     FFMPEG:                      YES
--       avcodec:                   YES (57.107.100)
--       avformat:                  YES (57.83.100)
--       avutil:                    YES (55.78.100)
--       swscale:                   YES (4.8.100)
--       avresample:                YES (3.7.0)
--     GStreamer:                   NO
--     v4l/v4l2:                    YES (linux/videodev2.h)
--
--   Parallel framework:            pthreads
--
--   Trace:                         YES (with Intel ITT)
--
--   Other third-party libraries:
--     Intel IPP:                   2019.0.0 Gold [2019.0.0]
--            at:
/home/yang/Projects/opencv/build/3rdparty/ippicv/ippicv_lnx/icv
--     Intel IPP IW:                sources (2019.0.0)
--               at:
/home/yang/Projects/opencv/build/3rdparty/ippicv/ippicv_lnx/iw
--     Lapack:                      NO
--     Eigen:                       NO
--     Custom HAL:                  NO
--     Protobuf:                    build (3.5.1)
--
--   NVIDIA CUDA:                   YES (ver 10.2, CUFFT CUBLAS)
--     NVIDIA GPU arch:             61
--     NVIDIA PTX archs:
--
--   cuDNN:                         YES (ver 7.6.5)
--
--   OpenCL:                        YES (no extra features)
--     Include path:
/home/yang/Projects/opencv/3rdparty/include/opencl/1.2
--     Link libraries:              Dynamic load
--
--   Python 2:
--     Interpreter:                 /usr/bin/python2.7 (ver 2.7.17)
--     Libraries:                   /usr/lib/x86_64-linux-gnu/libpython2.7.so
(ver 2.7.17)
--     numpy:
/home/yang/.local/lib/python2.7/site-packages/numpy/core/include (ver 1.16.5)
--     install path:                lib/python2.7/dist-packages/cv2/python-2.7
--
--   Python 3:
--     Interpreter:                 /usr/bin/python3 (ver 3.6.9)
--     Libraries:                   /usr/lib/x86_64-linux-gnu/libpython3.6m.so
(ver 3.6.9)
--     numpy:
/home/yang/.local/lib/python3.6/site-packages/numpy/core/include (ver 1.18.0)
--     install path:                lib/python3.6/dist-packages/cv2/python-3.6
--
--   Python (for build):            /usr/bin/python2.7
--
--   Java:
--     ant:                         NO
--     JNI:                         NO
--     Java wrappers:               NO
--     Java tests:                  NO
--
--   Install to:                    /usr/local
-- -----------------------------------------------------------------
--
-- Configuring done
-- Generating done
-- Build files have been written to: /home/yang/Projects/opencv/build
```


## Test DNN with CUDA backend
```
export OPENCV_TEST_DATA_PATH=/home/yang/Projects/opencv_extra/testdata
./bin/opencv_test_dnn
```

## Results
```
[ RUN      ] Test_Caffe_nets.FasterRCNN_vgg16/0, where GetParam() = CUDA/CUDA
TEST ERROR: Don't use 'optional' findData() for dnn/VGG16_faster_rcnn_final.caffemodel
[       OK ] Test_Caffe_nets.FasterRCNN_vgg16/0 (601 ms)
[ RUN      ] Test_Caffe_nets.FasterRCNN_vgg16/1, where GetParam() = CUDA/CUDA_FP16
TEST ERROR: Don't use 'optional' findData() for dnn/VGG16_faster_rcnn_final.caffemodel
Unmatched prediction: class 2 score 0.941406 box [521.174 x 283.55 from (81.848, 165.531)]
/home/yang/Projects/opencv/modules/dnn/test/test_common.impl.hpp:130: Failure
Value of: matched
  Actual: false
Expected: true
model name: VGG16_faster_rcnn_final.caffemodel
Unmatched prediction: class 7 score 0.997070 box [230.097 x 83.9558 from (481.496, 91.4764)]
/home/yang/Projects/opencv/modules/dnn/test/test_common.impl.hpp:130: Failure
Value of: matched
  Actual: false
Expected: true
model name: VGG16_faster_rcnn_final.caffemodel
Unmatched prediction: class 12 score 0.980957 box [219.77 x 383.609 from (126.853, 182.158)]
/home/yang/Projects/opencv/modules/dnn/test/test_common.impl.hpp:130: Failure
Value of: matched
  Actual: false
Expected: true
model name: VGG16_faster_rcnn_final.caffemodel
Unmatched reference: class 2 score 0.949398 box [501.96 x 252.708 from (99.2454, 210.141)]
/home/yang/Projects/opencv/modules/dnn/test/test_common.impl.hpp:140: Failure
Expected: (refScores[i]) <= (confThreshold), actual: 0.949398 vs 0.8
model name: VGG16_faster_rcnn_final.caffemodel
Unmatched reference: class 7 score 0.997022 box [240.844 x 83.6312 from (481.841, 92.3218)]
/home/yang/Projects/opencv/modules/dnn/test/test_common.impl.hpp:140: Failure
Expected: (refScores[i]) <= (confThreshold), actual: 0.997022 vs 0.8
model name: VGG16_faster_rcnn_final.caffemodel
Unmatched reference: class 12 score 0.993028 box [217.773 x 373.789 from (133.221, 189.377)]
/home/yang/Projects/opencv/modules/dnn/test/test_common.impl.hpp:140: Failure
Expected: (refScores[i]) <= (confThreshold), actual: 0.993028 vs 0.8
model name: VGG16_faster_rcnn_final.caffemodel
[  FAILED  ] Test_Caffe_nets.FasterRCNN_vgg16/1, where GetParam() = CUDA/CUDA_FP16 (4722 ms)

...

[----------] Global test environment tear-down
[ SKIPSTAT ] 109 tests skipped
[ SKIPSTAT ] TAG='mem_6gb' skip 3 tests
[ SKIPSTAT ] TAG='verylong' skip 3 tests
[ SKIPSTAT ] TAG='dnn_skip_cuda' skip 46 tests
[ SKIPSTAT ] TAG='dnn_skip_cuda_fp16' skip 54 tests
[ SKIPSTAT ] TAG='skip_other' skip 3 tests
[==========] 1918 tests from 64 test cases ran. (129112 ms total)
[  PASSED  ] 1916 tests.
[  FAILED  ] 2 tests, listed below:
[  FAILED  ] Test_Caffe_nets.FasterRCNN_vgg16/1, where GetParam() = CUDA/CUDA_FP16
[  FAILED  ] Test_Caffe_layers.ROIPooling_Accuracy/1, where GetParam() = CUDA/CUDA_FP16

 2 FAILED TESTS
```


## Tracing using nvprof

Options:
```
nvprof --cpu-thread-tracing on --profile-child-processes --trace gpu,api --cpu-profiling on -o opencv_yolov3_%p.nvvp
```

## Visualize Profiling Results
```
nvvp -vm /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java
```

## Generate calling graph
```
~/Projects/opencv/build/graph$ cmake --graphviz=opencv . ..
```

## Test gapi


```
~/Projects/opencv/build$ make opencv_test_gapi opencv_perf_gapi -j15
~/Projects/opencv/build$ ./bin/opencv_test_gapi
[----------] Global test environment tear-down
[==========] 21103 tests from 371 test cases ran. (183006 ms total)
[  PASSED  ] 21102 tests.
[  FAILED  ] 1 test, listed below:
[  FAILED  ] GPU.Symm7x7_test

 1 FAILED TEST
  YOU HAVE 1005 DISABLED TESTS

~/Projects/opencv/build$ ./bin/opencv_perf_gapi
```

## GAPI example

```
~/Projects/opencv/build$ make example_gapi_*
```
Source code is in `opencv/modules/gapi/samples`.

## How to create a new backend

### Kernel implementation

See https://docs.opencv.org/master/d0/d25/gapi_kernel_api.html

Define macro like `GAPI_OCV_KERNEL` and `GAPI_OCL_KERNEL` to help define kernel
implementation, specifically the `run()` function, where the actual computation
sits.

The interfaces of these kernels are defined in
`modules/gapi/include/opencv2/gapi/*.hpp` files such as the `imgproc.hpp` and the
`core.hpp` in it.  These interfaces are defined using G_TYPED_KERNEL, a helper
macro.

The implementation of these kernels are provided by backends.  Taking the OCV
(OpenCV CPU) backend as an example, in
`modules/gapi/include/opencv2/gapi/cpu/*.hpp`, it declares `kernels()` functions
that return GKernelPackage, which includes a set of kernel implementations.  The
implementaion of this `kernels()` function is in
`modules/gapi/src/backends/cpu/gcpu*.cpp`.

Kernels can be used with G-API-supplied method `::on()`.  And this method is
usually wrapped to enable optional parameters.  For example, `Canny` wraps
`GCanny::on`.

### Backend implementation

#### Interfaces for OpenCV Applications

Factory function `cv::gapi::cpu::backend()` in `include/opencv2/gapi/cpu/gcpukernel.hpp`
Helper functions for making `cv::gapi::cpu::GOCVFunctor` (`cv::gapi::GFunctor`) in `include/opencv2/gapi/cpu/gcpukernel.hpp`
Class `GCPUKernel` in `include/opencv2/gapi/cpu/gcpukernel.hpp`
Class `GCPUContext` in `include/opencv2/gapi/cpu/gcpukernel.hpp`
Factory function `cv::gapi::imgproc::cpu::kernels()` in `include/opencv2/gapi/cpu/imgproc.hpp`
Factory function `cv::gapi::core::cpu::kernels()` in `include/opencv2/gapi/cpu/core.hpp`
Factory function `cv::gapi::video::cpu::kernels()` in `include/opencv2/gapi/cpu/video.hpp`
Functions in `detail` namespace seems unused.

An interface can be implemented by multiple backends.  How the backend is chosen
(indicated) at runtime is unclear (the document is not ready).  This needs
to be figured out.

## How does DNN module in OpenCV work

An DNN application needs to include only `opencv2/dnn.hpp`, which refers to
`modules/dnn/include/dnn/dnn.hpp`, a file that contains most DNN interfaces such
as `readNet`, the classess of `Net`, `Model`, and `Layer` (for customizing layers).

The `Net` class represents the neural network.  An application calls
`Net::forward()` to start forward calculation, which loops over layers' forward
function.  However, it is slightly different for CUDA backends, with which the
`forward` function of each `CUDABackendNode` is called instead of the layer's
forward function.  The `CUDABackendNode` is initialized in `initCUDA` in each
layer class.  This `initCUDA` is called in the `initCUDABackend` function of the
implementation structure `struct Net::Impl` of the Net class.

Ownership and calling graph:
Net::forward()  "modules/dnn/src/dnn.cpp"
  Net::Impl::setUpNet  "modules/dnn/src/dnn.cpp"
    Net::Impl::initBackend
      Net::Impl::initCUDABackend
        cv::dnn::LayerData::layerInstance  (Ptr<cv::dnn::Layer>
          cv::dnn::Layer::initCUDA  // initCUDA backend node for the layer
          // reserve memory for the layer
  Net::forwardToLayer()  "dnn.cpp"
    Net::forwardLayer()  "dnn.cpp"
      // for CUDA backend
      CUDABackendNode::forward()  -- CUDABackendNode is a base class; we use convolution as an example in the following procedure.  declared in "modules/dnn/src/op_cuda.hpp"
        cv::dnn::cuda4dnn::ConvolutionOp::forward() -- ConvolutionOp inherites CUDABandendNode.  "modules/dnn/src/cuda4dnn/primitives/convolution.hpp"
          cv::dnn::cuda4dnn::csl::Convolution::convolve()    "modules/dnn/src/cuda4dnn/csl/tensor_ops.hpp"
	    cv::dnn::cuda4dnn::csl::cudnn::convolve()  -- it wraps cudnnConvolutionForward function.   "modules/dnn/src/cuda4dnn/csl/cudnn/convolution.hpp"


Similarly, for pooling:
Net::forward()  "modules/dnn/src/dnn.cpp"
  Net::forwardToLayer()  "dnn.cpp"
    Net::forwardLayer()  "dnn.cpp"
      // for CUDA backend
      CUDABackendNode::forward()  -- CUDABackendNode is a base class; we use convolution as an example in the following procedure.  declared in "modules/dnn/src/op_cuda.hpp"
        cv::dnn::cuda4dnn::PoolingOp::forward() -- PoolingOp inherites CUDABandendNode.  "modules/dnn/src/cuda4dnn/primitives/pooling.hpp"
          cv::dnn::cuda4dnn::csl::Pooling::pool()    "modules/dnn/src/cuda4dnn/csl/tensor_ops.hpp"
	    cv::dnn::cuda4dnn::csl::cudnn::pool()  -- it wraps cudnnPoolingForward function.   "modules/dnn/src/cuda4dnn/csl/cudnn/pooling.hpp"


Slightly differently, for region:
Net::forward()  "modules/dnn/src/dnn.cpp"
  Net::forwardToLayer()  "dnn.cpp"
    Net::forwardLayer()  "dnn.cpp"
      // for CUDA backend
      CUDABackendNode::forward()  -- CUDABackendNode is a base class; we use convolution as an example in the following procedure.  declared in "modules/dnn/src/op_cuda.hpp"
        cv::dnn::cuda4dnn::RegionOp::forward() -- RegionOp inherites CUDABandendNode.  "modules/dnn/src/cuda4dnn/primitives/region.hpp"
          cv::dnn::cuda4dnn::kernels::region()    declared in "modules/dnn/src/cuda4dnn/kernels/region.hpp"; implemented in "modules/dnn/src/cuda/region.cu"

All layers are defined in `modules/dnn/include/opencv2/dnn/all_layers.hpp` and
implemented in separate files like `modules/dnn/src/layers/convolution_layer.cpp`.
Each layer is declared as `XXXLayer`, inheriting `Layer`, which is defined in
`modules/dnn/include/opencv2/dnn/dnn.hpp`.

### From DNN model and weights to Net graph

Taking Darknet YOLO as an example.

cv::dnn::Net::readNet "modules/dnn/src/dnn.cpp"
  cv::dnn::readNetFromDarknet // declared in "modules/dnn/include/opencv2/dnn/dnn.hpp", defined in "modules/dnn/src/darknet/darknet_importer.cpp"
    cv::dnn::::DarknetImporter::DarknetImporter "modules/dnn/src/darknet/darknet_importer.cpp"
      cv::dnn::::ReadNetParamsFromCfgStreamOrDie "modules/dnn/src/darknet/darknet_io.cpp", declared in its `hpp`.
      cv::dnn::::ReadNetParamsFromBinaryStreamOrDie "modules/dnn/src/darknet/darknet_io.cpp", declared in its `hpp`.


## How does OpenCV Graph API tests work

Test fixture is defined in `gapi/test/common/gapi_imgproc_tests.hpp`.
Value-parameterized tests are defined in corresponding `*_inl.hpp`, e.g.,
`gapi/test/common/gapi_imgproc_tests_inl.hpp`.

## How to combine DNN module and Graph API module

Try this:

In GAPI module:
* create dnn backend
* define convolution kernel
* implement convolution kernel
* implement test for the convolution kernel
* test

## Run gapi example with video

```
~/Projects/opencv/build$ ./bin/example_gapi_api_example ~/Projects/opencv_extra/testdata/gpu/video/768x576.avi
```

## Run YOLO

```
./bin/example_dnn_object_detection --config=/home/yang/Projects/opencv_extra/testdata/dnn/yolo-voc.cfg --model=/home/yang/Projects/opencv_extra/testdata/dnn/yolo-voc.weights --classes=/home/yang/Projects/opencv/samples/data/dnn/object_detection_classes_pascal_voc.txt --width=416 --height=416 --scale=0.00392 --input=/home/yang/Projects/opencv_extra/testdata/dnn/street.png --rgb
```
