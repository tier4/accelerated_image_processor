---
icon: simple/markdown
---

# accelerated_image_processor_compression

This package provides compression functionalities for images using various algorithms.
It also includes support for hardware acceleration on NVIDIA Jetson devices using the Jetson Multimedia API.

## Compressor Supports

| Type   | Compressor             | Backend                                                                             | Device |
| ------ | ---------------------- | ----------------------------------------------------------------------------------- | ------ |
| `JPEG` | `JetsonJPEGCompressor` | [jetsonJPEG](https://docs.nvidia.com/jetson/l4t-multimedia/classNvJPEGEncoder.html) | Jetson |
|        | `NvJPEGCompressor`     | [nvJPEG](https://developer.nvidia.com/nvjpeg)                                       | GPU    |
|        | `CpuJPEGCompressor`    | [TurboJPEG](https://github.com/libjpeg-turbo/libjpeg-turbo)                         | CPU    |

## Example Usage in ROS 2

The following code demonstrates how to leverage the compressor in your ROS 2 codebase:

```c++ title="Compressor Example in ROS 2"
#include <accelerated_image_processor_common/datatype.hpp>
#include <accelerated_image_processor_compression/builder.hpp>

#include <rclcpp/rclcpp.hpp>

using namespace accelerated_image_processor;

class SomeNode final : public rclcpp::Node
{
public:
  explicit SomeNode(const rclcpp::NodeOptions & options) : Node("some_node", options)
  {
    // Choose compression type
    compression::CompressionType type = compression::CompressionType::JPEG;
    compressor_ = compression::create_compressor<SomeNode, &SomeNode::publish>(type, this);

    // Update parameters of the compressor
    for (auto & [name, value] : compressor_->parameters()) {
      std::visit([&](auto & v) {
        using T = std::decay_t<decltype(v)>;
        v = this->declare_parameter<T>(name, v);
      }, value);
    }

    // Create a subscription and publisher
    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "~/input/image", 10, [this](const sensor_msgs::msg::Image::ConstSharedPtr msg) { this->callback(msg); });
    publisher_ = this->create_publisher<sensor_msgs::msg::CompressedImage>("~/output/image", 10);
  }

private:
  void callback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
  {
    common::Image image;
    // Convert the message to image...
    compressor_->process(image);
  }

  void publish(const common::Image & image)
  {
    sensor_msgs::msg::CompressedImage msg;
    // Convert the image to message...
    publisher_->publish(msg);
  }

  std::unique_ptr<compression::Compressor> compressor_; //!< Compressor

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_; //!< Subscription
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr publisher_; //!< Publisher
};
```

## Unit Testings

```bash title="Build Unit Tests"
colcon build \
    --symlink-install \
    --packages-up-to accelerated_image_processor_compression
```

```bash title="Run Unit Tests"
colcon test \
    --event-handlers console_direct+ \
    --packages-select accelerated_image_processor_compression
```
