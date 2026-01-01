# accelerated_image_processor_pipeline

This package provides functionalities for image processing pipelines using various libraries.

## Processor Supports

| Task          | Processor              | Backend                                                                            | Device |
| ------------- | ---------------------- | ---------------------------------------------------------------------------------- | ------ |
| Rectification | `NppRectifier`         | [NVIDIA Performance Primitives (NPP)](https://docs.nvidia.com/cuda/npp/index.html) | GPU    |
|               | `OpencCvCudaRectifier` | [OpenCV CUDA](https://opencv.org/platforms/cuda/)                                  | GPU    |
|               | `CpuRectifier`         | [OpenCV](https://opencv.org/)                                                      | CPU    |

## Example Usage in ROS 2

The following code demonstrates how to use each processor in your ROS 2 codebase.

### Rectification

```c++
#include <accelerated_image_processor_common/datatype.hpp>
#include <accelerated_image_processor_pipeline/builder.hpp>

#include <rclcpp/rclcpp.hpp>

using namespace accelerated_image_processor;

class SomeNode final : public rclcpp::Node
{
public:
  explicit SomeNode(const rclcpp::NodeOptions & options) : Node("some_node", options)
  {
    rectifier_ = pipeline::create_rectifier<SomeNode, &SomeNode::publish>(this);

    // Update parameters of the compressor
    for (auto & [name, value] : compressor_->parameters()) {
      std::visit([&](auto & v) {
        using T = std::decay_t<decltype(v)>;
        v = this->declare_parameter<T>(name, v);
      }, value);
    }

    // Create a subscription and publisher
    image_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "~/input/image", 10, [this](const sensor_msgs::msg::Image::ConstSharedPtr msg) { this->image_callback(msg); });
    camera_info_subscription_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      "~/input/camera_info", 10, [this](const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) { this->camera_info_callback(msg); });
    image_publisher_ = this->create_publisher<sensor_msgs::msg::CompressedImage>("~/output/image", 10);
    camera_info_publisher_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("~/output/camera_info", 10);
  }

private:
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
  {
    common::Image image;
    // Convert the message to image...
    if (rectifier_->is_ready()) {
      rectifier_->process(image);
    }
  }

  void camera_info_callback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg)
  {
    rectifier_->set_camera_info(msg);
  }

  void publish(const common::Image & image)
  {
    const common::CameraInfo & camera_info = rectifier_->get_camera_info();

    sensor_msgs::msg::CompressedImage image_msg;
    sensor_msgs::msg::CameraInfo camera_info_msg;
    // Convert the image and camera info to message...
    image_publisher_->publish(image_msg);
    camera_info_publisher_->publish(camera_info_msg);
  }

  std::unique_ptr<pipeline::Rectifier> rectifier_; //!< Rectifier

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_; //!< Image subscription
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_subscription_; //!< CameraInfo subscription
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr image_publisher_; //!< Image publisher
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_publisher_; //!< CameraInfo publisher
};
```
