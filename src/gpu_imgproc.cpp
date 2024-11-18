#include "gpu_imgproc/gpu_imgproc.hpp"

#include <future>

namespace gpu_imgproc {

GpuImgProc::GpuImgProc(const rclcpp::NodeOptions & options)
    : Node("gpu_imgproc", options), rectifier_active_(false) {
    RCLCPP_INFO(this->get_logger(), "Initializing node gpu_imgproc");

    // std::string image_raw_topic = this->declare_parameter<std::string>("image_raw_topic", "/camera/image_raw");
    // std::string camera_info_topic = this->declare_parameter<std::string>("camera_info_topic", "/camera/camera_info");
    // std::string image_rect_topic = this->declare_parameter<std::string>("image_rect_topic", "/camera/image_rect");
    std::string rect_impl = this->declare_parameter<std::string>("rect_impl", "npp");
    bool use_opencv_map_init = this->declare_parameter<bool>("use_opencv_map_init", false);
    alpha_ = this->declare_parameter<double>("alpha", 0.0);
    jpeg_quality_ = this->declare_parameter<int32_t>("jpeg_quality", 60);
    bool do_rectify = this->declare_parameter<bool>("do_rectify", true);
    type_adaptation_active_ = this->declare_parameter<bool>("type_adaption_active", true);

    // RCLCPP_INFO(this->get_logger(), "Subscribing to %s", image_raw_topic.c_str());
    // RCLCPP_INFO(this->get_logger(), "Subscribing to %s", camera_info_topic.c_str());
    // RCLCPP_INFO(this->get_logger(), "Publishing to %s", image_rect_topic.c_str());

    std::string available_impls = "";
#ifdef NPP_AVAILABLE
    available_impls += "npp";
#endif
#ifdef OPENCV_AVAILABLE
    if (available_impls != "") {
        available_impls += ", ";
    }
    available_impls += "opencv_cpu";
#endif
#ifdef OPENCV_CUDA_AVAILABLE
    if (available_impls != "") {
        available_impls += ", ";
    }
    available_impls += "opencv_gpu";
#endif

    if (available_impls == "") {
        RCLCPP_ERROR(this->get_logger(),
        "No rectification implementations available. Please make sure that at least one of the following libraries is installed:\n"
        "- OpenCV\n"
        "- OpenCV CUDA\n"
        "- NVIDIA Performance Primitives\n");
        return;
    }

    if (0) {
#ifdef NPP_AVAILABLE
    } else if (rect_impl == "npp") {
        RCLCPP_INFO(this->get_logger(), "Using NPP implementation for rectification");
        rectifier_impl_ = Rectifier::Implementation::NPP;
#endif
#ifdef OPENCV_AVAILABLE
    } else if (rect_impl == "opencv_cpu") {
        RCLCPP_INFO(this->get_logger(), "Using CPU OpenCV implementation for rectification");
        rectifier_impl_ = Rectifier::Implementation::OpenCV_CPU;
#endif
#ifdef OPENCV_CUDA_AVAILABLE
    } else if (rect_impl == "opencv_gpu") {
        RCLCPP_INFO(this->get_logger(), "Using GPU OpenCV implementation for rectification");
        rectifier_impl_ = Rectifier::Implementation::OpenCV_GPU;
#endif
    } else {
        RCLCPP_ERROR(this->get_logger(), "Invalid implementation: %s. Available options: %s", rect_impl.c_str(), available_impls.c_str());
        return;
    }

    if (use_opencv_map_init) {
        RCLCPP_INFO(this->get_logger(), "Using OpenCV map initialization");
        mapping_impl_ = Rectifier::MappingImpl::OpenCV;
    } else {
        RCLCPP_INFO(this->get_logger(), "Using Non-OpenCV map initialization");
        mapping_impl_ = Rectifier::MappingImpl::NPP;
    }

#ifdef JETSON_AVAILABLE
    raw_compressor_ = std::make_shared<JpegCompressor::JetsonCompressor>("raw_compressor");
    rect_compressor_ = std::make_shared<JpegCompressor::JetsonCompressor>("rect_compressor");
#elif NVJPEG_AVAILABLE
    raw_compressor_ = std::make_shared<JpegCompressor::NVJPEGCompressor>();
    rect_compressor_ = std::make_shared<JpegCompressor::NVJPEGCompressor>();
#elif TURBOJPEG_AVAILABLE
    raw_compressor_ = std::make_shared<JpegCompressor::CPUCompressor>();
    rect_compressor_ = std::make_shared<JpegCompressor::CPUCompressor>();
#else
    RCLCPP_ERROR(this->get_logger(), "No JPEG compressor available");
    return;
#endif

    // Query QoS using timer to adapt to the case of image publisher starts after this consturctor
    qos_request_timer_ = rclcpp::create_timer(this, this->get_clock(), std::chrono::milliseconds(100),
                                              [do_rectify, this]()
                                              {this->determineQosCallback(do_rectify);});
}

GpuImgProc::~GpuImgProc() {
    RCLCPP_INFO(this->get_logger(), "Shutting down node gpu_imgproc");
}

void GpuImgProc::determineQosCallback(bool do_rectify) {
    // Query QoS to publisher to align the QoS for the topics to be published
    auto get_qos =  [this](std::string& topic_name, rclcpp::QoS& qos) -> bool {
        auto qos_list = this->get_publishers_info_by_topic(topic_name);
        if (qos_list.size() < 1) {
            RCLCPP_INFO_STREAM(this->get_logger(), "Waiting for" << topic_name << " ...");
            return false;
        } else if (qos_list.size() > 1) {
            RCLCPP_ERROR(this->get_logger(),
                         "Multiple publisher for %s are detected. Cannot determine proper QoS",
                         topic_name);
            return false;
        } else {
            RCLCPP_INFO_STREAM(this->get_logger(),
                               "QoS for " << topic_name << " is acquired.");
            qos = qos_list[0].qos_profile();
            return true;
        }
    };

    std::string img_sub_topic_name = this->get_node_topics_interface()->resolve_topic_name(
        "image_raw", false);
    std::string info_sub_topic_name = this->get_node_topics_interface()->resolve_topic_name(
        "camera_info", false);

    // Query QoS to publisher to align the QoS for the topics to be published
    rclcpp::QoS img_qos(1);
    if (!get_qos(img_sub_topic_name, img_qos)) {
        // Publisher is not ready yet
        return;
    }

    rclcpp::QoS info_qos(1);
    if (do_rectify) {
        if (!get_qos(info_sub_topic_name, info_qos)) {
            // Publisher is not ready yet
            return;
        }
    }

    if (type_adaptation_active_)
    {
        gpu_image_sub_ = this->create_subscription<ImageContainer>(
            img_sub_topic_name, img_qos, std::bind(&GpuImgProc::gpuImageCallback, this, std::placeholders::_1));
    }
    else {
        img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            img_sub_topic_name, img_qos, std::bind(&GpuImgProc::imageCallback, this, std::placeholders::_1));
    }
    compressed_pub_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(
            "image_raw/compressed", img_qos);
    

    if (do_rectify) {
      if (type_adaptation_active_)
      {
        gpu_rectified_pub_ = this->create_publisher<ImageContainer>(
            "image_rect", img_qos);
      }
      else
      {
        rectified_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
          "image_rect", img_qos);
      }
      
      rect_compressed_pub_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(
          "image_rect/compressed", img_qos);

      info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
          info_sub_topic_name, info_qos, std::bind(&GpuImgProc::cameraInfoCallback, this, std::placeholders::_1));
    }

    // Once all queries receive sufficient results, stop the timer
    qos_request_timer_->cancel();
}

void GpuImgProc::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    RCLCPP_DEBUG(this->get_logger(), "Received image");

    std::future<void> rectified_msg;
    if (rectifier_active_) {
        RCLCPP_DEBUG(this->get_logger(), "Rectifying image");
        rectified_msg =
            std::async(std::launch::async, [this, msg]() {
                sensor_msgs::msg::Image::UniquePtr rect_img;
                sensor_msgs::msg::CompressedImage::UniquePtr rect_comp_img;
                if (false) {
#ifdef NPP_AVAILABLE
                } else if (rectifier_impl_ == Rectifier::Implementation::NPP) {
                    rect_img = npp_rectifier_->rectify(*msg);
                    rect_comp_img = rect_compressor_->compress(*rect_img, jpeg_quality_);
#endif
#ifdef OPENCV_AVAILABLE
                } else if (rectifier_impl_ == Rectifier::Implementation::OpenCV_CPU) {
                    rect_img = cv_cpu_rectifier_->rectify(*msg);
                    rect_comp_img = rect_compressor_->compress(*rect_img, jpeg_quality_);
#endif
#ifdef OPENCV_CUDA_AVAILABLE
                } else if (rectifier_impl_ == Rectifier::Implementation::OpenCV_GPU) {
                    rect_img = cv_gpu_rectifier_->rectify(*msg);
                    rect_comp_img = rect_compressor_->compress(*rect_img, jpeg_quality_);
#endif
                } else {
                    RCLCPP_ERROR(this->get_logger(), "Invalid implementation");
                    return;
                }
                // XXX: As of 2023/Nov, publishing the topic via unique_ptr here may cause
                // SIGSEGV during cyclonedds process, so the topics are published via passing by value.
                // If this SIGSEGV issue will be resolved somehow, it's better to switch back to
                // publishing topics via unique_ptr for more efficiency.

                // rectified_pub_->publish(std::move(rect_img));
                // rect_compressed_pub_->publish(std::move(rect_comp_img));
                rectified_pub_->publish(*rect_img);
                rect_compressed_pub_->publish(*rect_comp_img);
            });
    } else {
        RCLCPP_DEBUG(this->get_logger(), "Not rectifying image");
    }

    std::future<void> compressed_msg =
            std::async(std::launch::async, [this, msg]() {
                sensor_msgs::msg::CompressedImage::UniquePtr comp_img;
                comp_img = raw_compressor_->compress(*msg, jpeg_quality_);
                // XXX: As of 2023/Nov, publishing the topic via unique_ptr here may cause
                // SIGSEGV during cyclonedds process, so the topics are published via passing by value.
                // If this SIGSEGV issue will be resolved somehow, it's better to switch back to
                // publishing topics via unique_ptr for more efficiency.

                // compressed_pub_->publish(std::move(comp_img));
                compressed_pub_->publish(*comp_img);
            });

    if (rectifier_active_) {
        rectified_msg.wait();
    }
    compressed_msg.wait();
}

void GpuImgProc::gpuImageCallback(const std::shared_ptr<ImageContainer> msg) {
    RCLCPP_DEBUG(this->get_logger(), "Received image");

    std::future<void> rectified_msg;
    if (rectifier_active_) {
        RCLCPP_DEBUG(this->get_logger(), "Rectifying image");
        rectified_msg =
            std::async(std::launch::async, [this, msg]() {
                ImageContainerUniquePtr rect_img;
                sensor_msgs::msg::CompressedImage::UniquePtr rect_comp_img;
                if (false) {
#ifdef NPP_AVAILABLE
                } else if (rectifier_impl_ == Rectifier::Implementation::NPP) {
                    rect_img = npp_rectifier_->rectify(*msg);
                    rect_comp_img = rect_compressor_->compress(*rect_img, jpeg_quality_);
#endif
#ifdef OPENCV_AVAILABLE
                } else if (rectifier_impl_ == Rectifier::Implementation::OpenCV_CPU) {
                    rect_img = cv_cpu_rectifier_->rectify(*msg);
                    rect_comp_img = rect_compressor_->compress(*rect_img, jpeg_quality_);
#endif
#ifdef OPENCV_CUDA_AVAILABLE
                } else if (rectifier_impl_ == Rectifier::Implementation::OpenCV_GPU) {
                    rect_img = cv_gpu_rectifier_->rectify(*msg);
                    rect_comp_img = rect_compressor_->compress(*rect_img, jpeg_quality_);
#endif
                } else {
                    RCLCPP_ERROR(this->get_logger(), "Invalid implementation");
                    return;
                }
                // XXX: As of 2023/Nov, publishing the topic via unique_ptr here may cause
                // SIGSEGV during cyclonedds process, so the topics are published via passing by value.
                // If this SIGSEGV issue will be resolved somehow, it's better to switch back to
                // publishing topics via unique_ptr for more efficiency.

                gpu_rectified_pub_->publish(std::move(rect_img));
                rect_compressed_pub_->publish(std::move(rect_comp_img));
            });
    } else {
        RCLCPP_DEBUG(this->get_logger(), "Not rectifying image");
    }

    std::future<void> compressed_msg =
            std::async(std::launch::async, [this, msg]() {
                sensor_msgs::msg::CompressedImage::UniquePtr comp_img;
                comp_img = raw_compressor_->compress(*msg, jpeg_quality_);
                // XXX: As of 2023/Nov, publishing the topic via unique_ptr here may cause
                // SIGSEGV during cyclonedds process, so the topics are published via passing by value.
                // If this SIGSEGV issue will be resolved somehow, it's better to switch back to
                // publishing topics via unique_ptr for more efficiency.

                compressed_pub_->publish(std::move(comp_img));
            });

    if (rectifier_active_) {
        rectified_msg.wait();
    }
    compressed_msg.wait();
}


void GpuImgProc::cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received camera info");

    if (msg->d.size() == 0 || msg->p.size() == 0) {
        RCLCPP_ERROR(this->get_logger(), "Camera info message does not contain distortion or projection matrix");
        return;
    }

    switch(rectifier_impl_) {
        case Rectifier::Implementation::NPP:
#if NPP_AVAILABLE
            RCLCPP_INFO(this->get_logger(), "Initializing NPP rectifier");
            npp_rectifier_ = std::make_shared<Rectifier::NPPRectifier>(*msg, mapping_impl_, alpha_);
            if (npp_rectifier_) {
                RCLCPP_INFO(this->get_logger(), "Initialized NPP rectifier");
                rectifier_active_ = true;
#if JETSON_AVAILABLE || NVJPEG_AVAILABLE
                // Use the same stream for rectifier
                // because compression process depends on rectified result
                auto stream = npp_rectifier_->GetCudaStream();
                rect_compressor_->setCudaStream(stream);
#endif
            } else {
                RCLCPP_ERROR(this->get_logger(), "Failed to initialize NPP rectifier");
                return;
            }
            break;
#else
            RCLCPP_ERROR(this->get_logger(), "NPP not enabled");
            return;
#endif
        case Rectifier::Implementation::OpenCV_CPU:
#ifdef OPENCV_AVAILABLE
            RCLCPP_INFO(this->get_logger(), "Initializing OpenCV CPU rectifier");
            cv_cpu_rectifier_ = std::make_shared<Rectifier::OpenCVRectifierCPU>(*msg, mapping_impl_, alpha_);
            if (cv_cpu_rectifier_) {
                RCLCPP_INFO(this->get_logger(), "Initialized OpenCV CPU rectifier");
                rectifier_active_ = true;
            } else {
                RCLCPP_ERROR(this->get_logger(), "Failed to initialize OpenCV rectifier");
                return;
            }
            break;
#else
            RCLCPP_ERROR(this->get_logger(), "OpenCV not enabled");
            return;
#endif
        case Rectifier::Implementation::OpenCV_GPU:
#ifdef OPENCV_CUDA_AVAILABLE
            RCLCPP_INFO(this->get_logger(), "Initializing OpenCV GPU rectifier");
            cv_gpu_rectifier_ = std::make_shared<Rectifier::OpenCVRectifierGPU>(*msg, mapping_impl_, alpha_);
            if (cv_gpu_rectifier_) {
                RCLCPP_INFO(this->get_logger(), "Initialized OpenCV GPU rectifier");
                rectifier_active_ = true;
            } else {
                RCLCPP_ERROR(this->get_logger(), "Failed to initialize OpenCV rectifier");
                return;
            }
            break;
#else
            RCLCPP_ERROR(this->get_logger(), "OpenCV CUDA not enabled");
            return;
#endif
        default:
            RCLCPP_ERROR(this->get_logger(), "Invalid rectifier implementation");
            return;
    }

    if (rectifier_active_) {
        // unsubscribe
        info_sub_.reset();
    }
}
} // namespace gpu_imgproc

#include <rclcpp_components/register_node_macro.hpp>

RCLCPP_COMPONENTS_REGISTER_NODE(gpu_imgproc::GpuImgProc)
