#pragma once

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include "cuda_blackboard/cuda_image.hpp"
#include "cuda_blackboard/cuda_unique_ptr.hpp"

#ifdef OPENCV_AVAILABLE
#include <opencv2/core.hpp>
#endif
#ifdef OPENCV_CUDA_AVAILABLE
#include <opencv2/core/cuda.hpp>
#endif

#if NPP_AVAILABLE
#include <nppdefs.h>
#endif

using CameraInfo = sensor_msgs::msg::CameraInfo;
using Image = sensor_msgs::msg::Image;

namespace Rectifier {

enum class Implementation {
    NPP,
    OpenCV_CPU,
    OpenCV_GPU
};

enum class MappingImpl {
    NPP,
    OpenCV
};

#if NPP_AVAILABLE
using CudaImage = cuda_blackboard::CudaImage;
class NPPRectifier {
public:
    cudaStream_t stream_;

    NPPRectifier(int width, int height,
                 const Npp32f *map_x, const Npp32f *map_y);
    NPPRectifier(const CameraInfo &info,
                 MappingImpl impl = MappingImpl::NPP,
                 double alpha = 0.0);
    ~NPPRectifier();
    cudaStream_t& GetCudaStream() {return stream_;}

    std::shared_ptr<CudaImage> rectify(const CudaImage  &msg);
private:
    Npp32f *pxl_map_x_;
    Npp32f *pxl_map_y_;
    int pxl_map_x_step_;
    int pxl_map_y_step_;
    int interpolation_;
    Npp8u *src_;
    Npp8u *dst_;
    int src_step_;
    int dst_step_;
};
#endif

#ifdef OPENCV_AVAILABLE
class OpenCVRectifierCPU {
public:
    OpenCVRectifierCPU(const CameraInfo &info,
                       MappingImpl impl = MappingImpl::OpenCV,
                       double alpha = 0.0);
    ~OpenCVRectifierCPU();

    Image::UniquePtr rectify(const Image &msg);
private:
    cv::Mat map_x_;
    cv::Mat map_y_;
    cv::Mat camera_intrinsics_;
    cv::Mat distortion_coeffs_;
};
#endif

#ifdef OPENCV_CUDA_AVAILABLE
class OpenCVRectifierGPU {
public:
    OpenCVRectifierGPU(const CameraInfo &info,
                       MappingImpl impl = MappingImpl::OpenCV,
                       double alpha = 0.0);
    ~OpenCVRectifierGPU();

    Image::UniquePtr rectify(const Image &msg);
private:
    cv::cuda::GpuMat map_x_;
    cv::cuda::GpuMat map_y_;
};
#endif

} // namespace Rectifier
