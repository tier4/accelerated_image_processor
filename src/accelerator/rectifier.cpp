#include "accelerator/rectifier.hpp"
#include <rclcpp/rclcpp.hpp>

#if NPP_AVAILABLE
#include <npp.h>
#include <nppi_data_exchange_and_initialization.h>
#include <nppi_geometry_transforms.h>
#include <nppi_support_functions.h>
#endif

#ifdef OPENCV_AVAILABLE
#include <image_geometry/pinhole_camera_model.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs.hpp>

#ifdef OPENCV_CUDA_AVAILABLE
// #include <opencv2/cudafeatures2d.hpp>
// #include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/core.hpp>
#endif
#endif


#define CHECK_NPP(status) \
    if (status != NPP_SUCCESS) {                                        \
        std::cerr << "NPP error: " << status                            \
                  << " (" <<  __FILE__ << ":" << __LINE__ << ")" << std::endl; \
    }

#define CHECK_CUDA(status) \
    if (status != cudaSuccess) {                                        \
        std::cerr << "CUDA error: " << cudaGetErrorName(status)         \
                  << " (" << __FILE__ << ":" <<  __LINE__ << ")" << std::endl; \
    }

namespace Rectifier {

static void compute_maps(int width, int height, const double *D, const double *P,
                  float *map_x, float *map_y) {
    std::cout <<  "No support for alpha in non-OpenCV mapping" << std::endl;

    double fx = P[0];
    double fy = P[5];
    double cx = P[2];
    double cy = P[6];

    double k1 = D[0];
    double k2 = D[1];
    double p1 = D[2];
    double p2 = D[3];
    double k3 = D[4];

    for (int v = 0; v < height; v++) {
        for (int u = 0; u < width; u++) {
            double x = (u - cx) / fx;
            double y = (v - cy) / fy;
            double r2 = x * x + y * y;
            double r4 = r2 * r2;
            double r6 = r4 * r2;
            double cdist = 1 + k1 * r2 + k2 * r4 + k3 * r6;
            double xd = x * cdist;
            double yd = y * cdist;
            double x2 = xd * xd;
            double y2 = yd * yd;
            double xy = xd * yd;
            double kr = 1 + p1 * r2 + p2 * r4;
            map_x[v * width + u] = (float)(fx * (xd * kr + 2 * p1 * xy + p2 * (r2 + 2 * x2)) + cx);
            map_y[v * width + u] = (float)(fy * (yd * kr + p1 * (r2 + 2 * y2) + 2 * p2 * xy) + cy);
        }
    }
}

#ifdef OPENCV_AVAILABLE
static void compute_maps_opencv(const CameraInfo &info, float *map_x, float *map_y, double alpha = 0.0) {
    cv::Mat camera_intrinsics(3, 3, CV_64F);
    cv::Mat distortion_coefficients(1, info.d.size(), CV_64F);

    for (int row=0; row<3; row++) {
        for (int col=0; col<3; col++) {
            camera_intrinsics.at<double>(row, col) = info.k[row * 3 + col];
        }
    }

    for (std::size_t col=0; col<info.d.size(); col++) {
        distortion_coefficients.at<double>(col) = info.d[col];
    }

    cv::Mat new_intrinsics = cv::getOptimalNewCameraMatrix(camera_intrinsics,
        distortion_coefficients,
        cv::Size(info.width, info.height),
        alpha);

    cv::Mat m1(info.height, info.width, CV_32FC1, map_x);
    cv::Mat m2(info.height, info.width, CV_32FC1, map_y);

    cv::initUndistortRectifyMap(camera_intrinsics,
        distortion_coefficients,
        cv::Mat(),
        new_intrinsics,
        cv::Size(info.width, info.height),
        CV_32FC1,
        m1, m2);
}
#endif

#if NPP_AVAILABLE
NPPRectifier::NPPRectifier(int width, int height,
                           const Npp32f *map_x, const Npp32f *map_y)
    : pxl_map_x_(nullptr), pxl_map_y_(nullptr) {
    cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    nppSetStream(stream_);

    pxl_map_x_ = nppiMalloc_32f_C1(width, height, &pxl_map_x_step_);
    if (pxl_map_x_ == nullptr) {
        std::cerr <<  "Failed to allocate GPU memory" << std::endl;
        return;
    }
    pxl_map_y_ = nppiMalloc_32f_C1(width, height, &pxl_map_y_step_);
    if (pxl_map_y_ == nullptr) {
        std::cerr <<  "Failed to allocate GPU memory" << std::endl;
        return;
    }

    src_ = nppiMalloc_8u_C3(width, height, &src_step_);
    if (src_ == nullptr) {
        std::cerr <<  "Failed to allocate GPU memory" << std::endl;
      return;
    }
    dst_ = nppiMalloc_8u_C3(width, height, &dst_step_);
    if (dst_ == nullptr) {
        std::cerr <<  "Failed to allocate GPU memory" << std::endl;
      return;
    }

    CHECK_CUDA(cudaMemcpy2DAsync(pxl_map_x_, pxl_map_x_step_, map_x, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice, stream_));
    CHECK_CUDA(cudaMemcpy2DAsync(pxl_map_y_, pxl_map_y_step_, map_y, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice, stream_));
}

NPPRectifier::NPPRectifier(const CameraInfo& info, MappingImpl impl, double alpha) {
    cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);

    nppSetStream(stream_);

    pxl_map_x_ = nppiMalloc_32f_C1(info.width, info.height, &pxl_map_x_step_);
    if (pxl_map_x_ == nullptr) {
        std::cerr <<  "Failed to allocate GPU memory" << std::endl;
        return;
    }
    pxl_map_y_ = nppiMalloc_32f_C1(info.width, info.height, &pxl_map_y_step_);
    if (pxl_map_y_ == nullptr) {
        std::cerr <<  "Failed to allocate GPU memory" << std::endl;
        return;
    }

    src_ = nppiMalloc_8u_C3(info.width, info.height, &src_step_);
    if (src_ == nullptr) {
        std::cerr << "Failed to allocate GPU memory" << std::endl;
      return;
    }
    dst_ = nppiMalloc_8u_C3(info.width, info.height, &dst_step_);
    if (dst_ == nullptr) {
        std::cerr << "Failed to allocate GPU memory" << std::endl;
      return;
    }

    std::cout << "Rectifying image with " << info.width << "x" << info.height << " pixels" << std::endl;

    // Create rectification map
    // TODO: Verify this works
    float *map_x = new float[info.width * info.height];
    float *map_y = new float[info.width * info.height];

#ifdef OPENCV_AVAILABLE
    if (impl == MappingImpl::OpenCV)
        compute_maps_opencv(info, map_x, map_y, alpha);
    else
#endif
    compute_maps(info.width, info.height,
                 info.d.data(), info.p.data(),
                 map_x, map_y);

    std::cout << "Copying rectification map to GPU" << std::endl;

    CHECK_CUDA(cudaMemcpy2DAsync(pxl_map_x_, pxl_map_x_step_, map_x, info.width * sizeof(float), info.width * sizeof(float), info.height, cudaMemcpyHostToDevice, stream_));
    CHECK_CUDA(cudaMemcpy2DAsync(pxl_map_y_, pxl_map_y_step_, map_y, info.width * sizeof(float), info.width * sizeof(float), info.height, cudaMemcpyHostToDevice, stream_));

    delete[] map_x;
    delete[] map_y;
}

NPPRectifier::~NPPRectifier() {
    if (pxl_map_x_ != nullptr) {
        nppiFree(pxl_map_x_);
    }

    if (pxl_map_y_ != nullptr) {
        nppiFree(pxl_map_y_);
    }

    if (src_ != nullptr) {
      nppiFree(src_);
    }

    if (dst_ != nullptr) {
      nppiFree(dst_);
    }

    cudaStreamDestroy(stream_);
}

Image::UniquePtr NPPRectifier::rectify(const Image &msg) {
    nppSetStream(stream_);
    Image::UniquePtr result = std::make_unique<Image>();
    result->header = msg.header;
    result->height = msg.height;
    result->width = msg.width;
    result->encoding = msg.encoding;
    result->is_bigendian = msg.is_bigendian;
    result->step = msg.step;

    result->data.resize(msg.data.size());

    NppiRect src_roi = {0, 0, (int)msg.width, (int)msg.height};
    NppiSize src_size = {(int)msg.width, (int)msg.height};
    NppiSize dst_roi_size = {(int)msg.width, (int)msg.height};

    CHECK_CUDA(cudaMemcpy2DAsync(src_, src_step_, msg.data.data(), msg.step, msg.width * 3, msg.height, cudaMemcpyHostToDevice, stream_));

    NppiInterpolationMode interpolation = NPPI_INTER_LINEAR;

    CHECK_NPP(nppiRemap_8u_C3R(
        src_, src_size, src_step_, src_roi,
        pxl_map_x_, pxl_map_x_step_, pxl_map_y_, pxl_map_y_step_,
        dst_, dst_step_, dst_roi_size, interpolation));

    CHECK_CUDA(cudaMemcpy2DAsync(static_cast<void*>(result->data.data()),
                                 result->step,
                                 static_cast<const void*>(dst_),
                                 dst_step_,
                                 msg.width * 3 * sizeof(Npp8u),  // in byte
                                 msg.height,
                                 cudaMemcpyDeviceToHost,
                                 stream_));

    // cv::Mat image(msg.height, msg.width, CV_8UC3, result->data.data(), result->step);
    // cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    // // save with timestamp
    // std::string filename = "rectified_" + std::to_string(msg.header.stamp.sec) + "_" + std::to_string(msg.header.stamp.nanosec) + ".png";
    // imwrite(filename, image);

    return result;
}
ImageContainerUniquePtr NPPRectifier::rectify(const ImageContainer &msg) {
    nppSetStream(msg.cuda_stream()->stream());
    ImageContainerUniquePtr result = std::make_unique<ImageContainer>(
        msg.header(), msg.height(), msg.width(), msg.encoding(), msg.step(), msg.cuda_stream()
    );
    
    NppiRect src_roi = {0, 0, (int)msg.width(), (int)msg.height()};
    NppiSize src_size = {(int)msg.width(), (int)msg.height()};
    NppiSize dst_roi_size = {(int)msg.width(), (int)msg.height()};

    NppiInterpolationMode interpolation = NPPI_INTER_LINEAR;

    // CHECK_CUDA(cudaMemcpy2DAsync(src_, src_step_, 
    //             msg.cuda_mem(), msg.step(), msg.width() * 3, 
    //             msg.height(), cudaMemcpyHostToDevice, msg.cuda_stream()->stream()));

    // CHECK_NPP(nppiRemap_8u_C3R(
    //     src_, src_size, src_step_, src_roi,
    //     pxl_map_x_, pxl_map_x_step_, pxl_map_y_, pxl_map_y_step_,
    //     dst_, dst_step_, dst_roi_size, interpolation));


    // CHECK_CUDA(cudaMemcpy2DAsync(static_cast<void*>(result->cuda_mem()),
    //                              result->step(),
    //                              static_cast<const void*>(dst_),
    //                              dst_step_,
    //                              msg.width() * 3 * sizeof(Npp8u),  // in byte
    //                              msg.height(),
    //                              cudaMemcpyDeviceToDevice,
    //                              result->cuda_stream()->stream()));

    CHECK_NPP(nppiRemap_8u_C3R(
        msg.cuda_mem(), src_size, msg.step(), src_roi,
        pxl_map_x_, pxl_map_x_step_, pxl_map_y_, pxl_map_y_step_,
        result->cuda_mem(), msg.step(), dst_roi_size, interpolation));
    return result;
}
#endif

#ifdef OPENCV_AVAILABLE
OpenCVRectifierCPU::OpenCVRectifierCPU(const CameraInfo &info, MappingImpl impl, double alpha) {
    map_x_ = cv::Mat(info.height, info.width, CV_32FC1);
    map_y_ = cv::Mat(info.height, info.width, CV_32FC1);

    if (impl == MappingImpl::OpenCV)
        compute_maps_opencv(info, map_x_.ptr<float>(), map_y_.ptr<float>(), alpha);
    else
        compute_maps(info.width, info.height,
                     info.d.data(), info.p.data(),
                     map_x_.ptr<float>(), map_y_.ptr<float>());
}

OpenCVRectifierCPU::~OpenCVRectifierCPU() {}

Image::UniquePtr OpenCVRectifierCPU::rectify(const Image &msg) {
    Image::UniquePtr result = std::make_unique<Image>();
    result->header = msg.header;
    result->height = msg.height;
    result->width = msg.width;
    result->encoding = msg.encoding;
    result->is_bigendian = msg.is_bigendian;
    result->step = msg.step;

    result->data.resize(msg.data.size());

    cv::Mat src(msg.height, msg.width, CV_8UC3, (void *)msg.data.data());
    cv::Mat dst(msg.height, msg.width, CV_8UC3, (void *)result->data.data());

    cv::remap(src, dst, map_x_, map_y_, cv::INTER_LINEAR);

    return result;
}

ImageContainerUniquePtr OpenCVRectifierCPU::rectify(const ImageContainer &msg) {
    RCLCPP_ERROR(
        rclcpp::get_logger("v4l2_camera"),
        "OpenCVRectifierCPU does not support dealing with image container on GPU");

    sensor_msgs::msg::Image image_msg;
    msg.get_sensor_msgs_image(image_msg);
    std::unique_ptr<sensor_msgs::msg::Image> image = rectify(image_msg);
    ImageContainerUniquePtr result = std::make_unique<ImageContainer>(std::move(image));
    return result;
}
#endif

#ifdef OPENCV_CUDA_AVAILABLE
OpenCVRectifierGPU::OpenCVRectifierGPU(const CameraInfo &info, MappingImpl impl, double alpha) {
    cv::Mat map_x(info.height, info.width, CV_32FC1);
    cv::Mat map_y(info.height, info.width, CV_32FC1);

    if (impl == MappingImpl::OpenCV)
        compute_maps_opencv(info, map_x.ptr<float>(), map_y.ptr<float>(), alpha);
    else
        compute_maps(info.width, info.height,
                     info.d.data(), info.p.data(),
                     map_x.ptr<float>(), map_y.ptr<float>());

    map_x_ = cv::cuda::GpuMat(map_x);
    map_y_ = cv::cuda::GpuMat(map_y);
}

OpenCVRectifierGPU::~OpenCVRectifierGPU() {}

Image::UniquePtr OpenCVRectifierGPU::rectify(const Image &msg) {
    Image::UniquePtr result = std::make_unique<Image>();
    result->header = msg.header;
    result->height = msg.height;
    result->width = msg.width;
    result->encoding = msg.encoding;
    result->is_bigendian = msg.is_bigendian;
    result->step = msg.step;

    result->data.resize(msg.data.size());

    cv::Mat src(msg.height, msg.width, CV_8UC3, (void *)msg.data.data());
    cv::cuda::GpuMat d_src = cv::cuda::GpuMat(src);
    cv::cuda::GpuMat d_dst = cv::cuda::GpuMat(cv::Size(msg.width, msg.height), src.type());

    cv::cuda::remap(d_src, d_dst, map_x_, map_y_, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

    // copy back to result
    cv::Mat dst(msg.height, msg.width, CV_8UC3, (void *)result->data.data());
    d_dst.download(dst);

    // cv::Mat image(msg.height, msg.width, CV_8UC3, result->data.data(), result->step);
    // cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    // // save with timestamp
    // std::string filename = "rectified_" + std::to_string(msg.header.stamp.sec) + "_" + std::to_string(msg.header.stamp.nanosec) + ".png";
    // imwrite(filename, image);

    return result;
}

ImageContainerUniquePtr OpenCVRectifierGPU::rectify(const ImageContainer &msg) {
    RCLCPP_ERROR(
        rclcpp::get_logger("v4l2_camera"),
        "OpenCVRectifierCPU does not support dealing with image container on GPU");

    sensor_msgs::msg::Image image_msg;
    msg.get_sensor_msgs_image(image_msg);
    std::unique_ptr<sensor_msgs::msg::Image> image = rectify(image_msg);
    ImageContainerUniquePtr result = std::make_unique<ImageContainer>(std::move(image));
    return result;
}
#endif

} // namespace Rectifier
