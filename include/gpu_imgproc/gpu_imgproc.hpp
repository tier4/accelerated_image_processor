#pragma once

#include <optional>
#include <rclcpp/rclcpp.hpp>
// #include <rcl_interfaces/msg/parameter.hpp>

#include "accelerator/rectifier.hpp"
#include "accelerator/jpeg_compressor.hpp"
// #include <sensor_msgs/msg/compressed_image.hpp>
#include "gpu_imgproc/task_queue.hpp"

namespace gpu_imgproc {

class GpuImgProc : public rclcpp::Node {
public:
    explicit GpuImgProc(const rclcpp::NodeOptions & options);
    virtual ~GpuImgProc();

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
    void determineQosCallback(bool do_rectify);

#if NPP_AVAILABLE
    std::shared_ptr<Rectifier::NPPRectifier> npp_rectifier_;
#endif
    std::shared_ptr<Rectifier::OpenCVRectifierCPU> cv_cpu_rectifier_;
#ifdef OPENCV_CUDA_AVAILABLE
    std::shared_ptr<Rectifier::OpenCVRectifierGPU> cv_gpu_rectifier_;
#endif
#ifdef JETSON_AVAILABLE
    std::shared_ptr<JpegCompressor::JetsonCompressor> raw_compressor_;
    std::shared_ptr<JpegCompressor::JetsonCompressor> rect_compressor_;
#elif NVJPEG_AVAILABLE
    std::shared_ptr<JpegCompressor::NVJPEGCompressor> raw_compressor_;
    std::shared_ptr<JpegCompressor::NVJPEGCompressor> rect_compressor_;
#elif TURBOJPEG_AVAILABLE
    std::shared_ptr<JpegCompressor::CPUCompressor> raw_compressor_;
    std::shared_ptr<JpegCompressor::CPUCompressor> rect_compressor_;
#endif

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rectified_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr rect_compressed_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr info_rect_pub_;

    rclcpp::TimerBase::SharedPtr qos_request_timer_;

    Rectifier::Implementation rectifier_impl_;
    bool rectifier_active_;
    double alpha_;
    int32_t jpeg_quality_;
    bool do_rectify_;

    size_t max_task_queue_length_;
    std::optional<util::TaskQueue> rectify_task_queue_;
    std::optional<std::thread> rectify_worker_;
    std::optional<util::TaskQueue> compress_task_queue_;
    std::optional<std::thread> compress_worker_;
};


} // namespace gpu_imgproc
