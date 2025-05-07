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
    bool use_opencv_map_init = this->declare_parameter<bool>("use_opencv_map_init", true);
    alpha_ = this->declare_parameter<double>("alpha", 0.0);
    jpeg_quality_ = this->declare_parameter<int32_t>("jpeg_quality", 60);
    do_rectify_ = this->declare_parameter<bool>("do_rectify", true);
    // NOTE: too short `max_task_queue_length_` may cause topic drop, while too large one may cause 100% system memory usage under many cameras/high framerate conditions
    max_task_queue_length_ = static_cast<size_t>(
        this->declare_parameter<int64_t>("max_task_queue_length", 5));

    // RCLCPP_INFO(this->get_logger(), "Subscribing to %s", image_raw_topic.c_str());
    // RCLCPP_INFO(this->get_logger(), "Subscribing to %s", camera_info_topic.c_str());
    // RCLCPP_INFO(this->get_logger(), "Publishing to %s", image_rect_topic.c_str());

    std::string available_impls = "";
#ifdef NPP_AVAILABLE
    available_impls += "npp";
#endif
    if (available_impls != "") {
        available_impls += ", ";
    }
    available_impls += "opencv_cpu";
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
    } else if (rect_impl == "opencv_cpu") {
        RCLCPP_INFO(this->get_logger(), "Using CPU OpenCV implementation for rectification");
        rectifier_impl_ = Rectifier::Implementation::OpenCV_CPU;
#ifdef OPENCV_CUDA_AVAILABLE
    } else if (rect_impl == "opencv_gpu") {
        RCLCPP_INFO(this->get_logger(), "Using GPU OpenCV implementation for rectification");
        rectifier_impl_ = Rectifier::Implementation::OpenCV_GPU;
#endif
    } else {
        RCLCPP_ERROR(this->get_logger(), "Invalid implementation: %s. Available options: %s", rect_impl.c_str(), available_impls.c_str());
        return;
    }

    if (!use_opencv_map_init) {
      RCLCPP_WARN(this->get_logger(),
                  "`use_opencv_map_init==false` is deprecated and no longer supported. "
                  "`use_opencv_map_init==true` is used automatically.");
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
                                              [this]()
                                              {this->determineQosCallback(do_rectify_);});
}

GpuImgProc::~GpuImgProc() {
    RCLCPP_INFO(this->get_logger(), "Shutting down node gpu_imgproc");
    if (do_rectify_) {
      rectify_task_queue_->stop();
      rectify_worker_->join();
    }
    compress_task_queue_->stop();
    compress_worker_->join();
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
                         topic_name.c_str());
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

    compressed_pub_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(
        "image_raw/compressed", img_qos);
    compress_task_queue_.emplace(max_task_queue_length_);
    compress_worker_.emplace(&util::TaskQueue::run, &compress_task_queue_.value());

    img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        img_sub_topic_name, img_qos, std::bind(&GpuImgProc::imageCallback, this, std::placeholders::_1));

    if (do_rectify) {
      rectified_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
          "image_rect", img_qos);
      rect_compressed_pub_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(
          "image_rect/compressed", img_qos);
      rectify_task_queue_.emplace(max_task_queue_length_);
      rectify_worker_.emplace(&util::TaskQueue::run, &rectify_task_queue_.value());

      info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
          info_sub_topic_name, info_qos, std::bind(&GpuImgProc::cameraInfoCallback, this, std::placeholders::_1));
      info_rect_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>(
          "camera_info_rect", info_qos);
    }

    // Once all queries receive sufficient results, stop the timer
    qos_request_timer_->cancel();
}

void GpuImgProc::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    RCLCPP_DEBUG(this->get_logger(), "Received image");

    JpegCompressor::ImageFormat image_format;
    if (msg->encoding == "rgb8") {
      image_format = JpegCompressor::ImageFormat::RGB;
    } else if (msg->encoding == "bgr8") {
      image_format = JpegCompressor::ImageFormat::BGR;
    } else {
      RCLCPP_ERROR_STREAM(this->get_logger(),
                          "Image encoding (" << msg->encoding << ") is not supported.");
    }

    if (rectifier_active_) {
        RCLCPP_DEBUG(this->get_logger(), "Rectifying image");
        rectify_task_queue_->addTask([this, msg, image_format]() {
                sensor_msgs::msg::Image::UniquePtr rect_img;
                sensor_msgs::msg::CompressedImage::UniquePtr rect_comp_img;
                sensor_msgs::msg::CameraInfo rect_info;
                if (false) {
#ifdef NPP_AVAILABLE
                } else if (rectifier_impl_ == Rectifier::Implementation::NPP) {
                    if (!npp_rectifier_->IsCameraInfoReady()) {
                        return;
                    }
                    rect_img = npp_rectifier_->rectify(*msg);
                    rect_comp_img = rect_compressor_->compress(*rect_img, jpeg_quality_, image_format);
                    rect_info = npp_rectifier_->GetCameraInfoRect();
#endif
                } else if (rectifier_impl_ == Rectifier::Implementation::OpenCV_CPU) {
                    if (!cv_cpu_rectifier_->IsCameraInfoReady()) {
                        return;
                    }
                    rect_img = cv_cpu_rectifier_->rectify(*msg);
                    rect_comp_img = rect_compressor_->compress(*rect_img, jpeg_quality_, image_format);
                    rect_info = cv_cpu_rectifier_->GetCameraInfoRect();
#ifdef OPENCV_CUDA_AVAILABLE
                } else if (rectifier_impl_ == Rectifier::Implementation::OpenCV_GPU) {
                    if (!cv_gpu_rectifier_->IsCameraInfoReady()) {
                        return;
                    }
                    rect_img = cv_gpu_rectifier_->rectify(*msg);
                    rect_comp_img = rect_compressor_->compress(*rect_img, jpeg_quality_, image_format);
                    rect_info = cv_gpu_rectifier_->GetCameraInfoRect();
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

                // updata header information camera info for rectified image
                rect_info.header = rect_img->header;
                info_rect_pub_->publish(std::move(rect_info));
            });
    } else {
        RCLCPP_DEBUG(this->get_logger(), "Not rectifying image");
    }

    compress_task_queue_->addTask([this, msg, image_format]() {
                sensor_msgs::msg::CompressedImage::UniquePtr comp_img;
                comp_img = raw_compressor_->compress(*msg, jpeg_quality_, image_format);
                // XXX: As of 2023/Nov, publishing the topic via unique_ptr here may cause
                // SIGSEGV during cyclonedds process, so the topics are published via passing by value.
                // If this SIGSEGV issue will be resolved somehow, it's better to switch back to
                // publishing topics via unique_ptr for more efficiency.

                // compressed_pub_->publish(std::move(comp_img));
                compressed_pub_->publish(*comp_img);
            });
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
            npp_rectifier_ = std::make_shared<Rectifier::NPPRectifier>(*msg, alpha_);
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
            RCLCPP_INFO(this->get_logger(), "Initializing OpenCV CPU rectifier");
            cv_cpu_rectifier_ = std::make_shared<Rectifier::OpenCVRectifierCPU>(*msg, alpha_);
            if (cv_cpu_rectifier_) {
                RCLCPP_INFO(this->get_logger(), "Initialized OpenCV CPU rectifier");
                rectifier_active_ = true;
            } else {
                RCLCPP_ERROR(this->get_logger(), "Failed to initialize OpenCV rectifier");
                return;
            }
            break;
        case Rectifier::Implementation::OpenCV_GPU:
#ifdef OPENCV_CUDA_AVAILABLE
            RCLCPP_INFO(this->get_logger(), "Initializing OpenCV GPU rectifier");
            cv_gpu_rectifier_ = std::make_shared<Rectifier::OpenCVRectifierGPU>(*msg, alpha_);
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
