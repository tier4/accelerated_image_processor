# accelerated_image_processor_decompression

This package provides decompression functionalities for compressed video streams using NVIDIA
NVDEC and NPP on CUDA-capable discrete GPUs.

## Decompressor Supports

| Decompressor             | Format                  | Backend | Device     |
| ------------------------ | ----------------------- | ------- | ---------- |
| `NvDecVideoDecompressor` | `VIDEO` (H264/H265/AV1) | NVDEC   | NVIDIA GPU |

> **Note**
> _Only a single hardware-accelerated backend is currently supported:_
> _`NvDecVideoDecompressor` directly loads `libnvcuvid` through the utility API provided by
> `nv-codec-headers`, then uses NPP to convert decoded frames to `BGR` on the GPU. The decompressor accepts a
> `common::Image` whose `format` field indicates the encoded format
> (`H264`, `H265`, or `AV1`). It outputs a `common::Image` with
> `format` set to `RAW`/`BGR`._

The package requires nv-codec-headers 13 or newer. If no local or system installation is found,
CMake downloads the pinned `n13.0.19.1` release, which requires NVIDIA driver 570.0 or newer on
Linux. Submit an empty `data` payload after the final encoded packet to drain delayed frames.

## Example Usage in ROS 2

Below is a minimal example of how to use the decompressor in a ROS 2 node.
It mirrors the example in the _compression_ package, but uses the
`decompression::create_decompressor` factory.

```c++
#include <accelerated_image_processor_common/datatype.hpp>
#include <accelerated_image_processor_decompression/builder.hpp>
#include <rclcpp/rclcpp.hpp>

using namespace accelerated_image_processor;

class SomeNode final : public rclcpp::Node
{
public:
  explicit SomeNode(const rclcpp::NodeOptions & options)
  : Node("some_node", options)
  {
    // Choose decompression type
    decompression::DecompressionType type = decompression::DecompressionType::VIDEO;
    decompressor_ = decompression::create_decompressor<SomeNode, &SomeNode::publish>(type, this);

    // Update parameters of the decompressor
    for (auto & [name, value] : decompressor_->parameters()) {
      std::visit([&](auto & v) {
        using T = std::decay_t<decltype(v)>;
        v = this->declare_parameter<T>(name, v);
      }, value);
    }

    // Subscription to compressed stream and publisher for raw image
    subscription_ = this->create_subscription<ffmpeg_image_transport_msgs::msg::FFMPEGPacket>(
      "~/input/image/compressed", 10,
      [this](const ffmpeg_image_transport_msgs::msg::FFMPEGPacket::ConstSharedPtr msg)
      { this->callback(msg); });
    publisher_ = this->create_publisher<sensor_msgs::msg::Image>("~/output/image", 10);
  }

private:
  void callback(const ffmpeg_image_transport_msgs::msg::FFMPEGPacket::ConstSharedPtr msg)
  {
    common::Image image;
    // Convert ROS message → accelerated_image_processor::common::Image …
    decompressor_->process(image);
  }

  void publish(const common::Image & image)
  {
    sensor_msgs::msg::Image msg;
    // Convert accelerated_image_processor::common::Image → ROS message …
    publisher_->publish(msg);
  }

  std::unique_ptr<decompression::Decompressor> decompressor_;
  rclcpp::Subscription<ffmpeg_image_transport_msgs::msg::FFMPEGPacket>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
};
```
