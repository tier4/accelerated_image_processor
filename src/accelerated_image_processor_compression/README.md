# accelerated_image_processor_compression

This package provides compression functionalities for images using various algorithms.
It also includes support for hardware acceleration on NVIDIA Jetson devices using the Jetson Multimedia API.

## Compressor Supports

| Compressor             | Format | Backend                                                                                  | Device |
| ---------------------- | ------ | ---------------------------------------------------------------------------------------- | ------ |
| `JetsonJPEGCompressor` | `JPEG` | [jetsonJPEG](https://docs.nvidia.com/jetson/l4t-multimedia/classNvJPEGEncoder.html)      | Jetson |
| `NvJPEGCompressor`     | `JPEG` | [nvJPEG](https://developer.nvidia.com/nvjpeg)                                            | GPU    |
| `CpuJPEGCompressor`    | `JPEG` | [TurboJPEG](https://github.com/libjpeg-turbo/libjpeg-turbo)                              | CPU    |
| `JetsonH264Compressor` | `H264` | [NvVideoEncoder](https://docs.nvidia.com/jetson/l4t-multimedia/classNvVideoEncoder.html) | Jetson |
| `JetsonH265Compressor` | `H265` | [NvVideoEncoder](https://docs.nvidia.com/jetson/l4t-multimedia/classNvVideoEncoder.html) | Jetson |
| `JetsonAV1Compressor`  | `AV1`  | [NvVideoEncoder](https://docs.nvidia.com/jetson/l4t-multimedia/classNvVideoEncoder.html) | Jetson |

## AV1 bitstream handling in `JetsonAV1Compressor`

### AV1 structure in a nutshell

An AV1 stream is a sequence of **OBUs** (Open Bitstream Units). Each OBU is a small
self-describing box: a 1-byte header (type + flags), an optional size field, and a payload.
The OBU types relevant here are:

| OBU                     | Role                                                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------------------------------- |
| Temporal Delimiter (TD) | Marks the beginning of a _temporal unit_ (≈ one packet carrying one displayed frame)                          |
| Sequence Header (SH)    | Stream-wide parameters (resolution, profile, bit depth, ...) the decoder needs before it can decode any frame |
| Frame                   | One coded picture (a key frame or an inter frame)                                                             |

Per the AV1 specification (Section 7.6.2), a decoder can start decoding mid-stream at a key
frame only when the _same_ packet also contains a sequence header OBU — such a packet is
called a _random access point_.

### What this package does

The Jetson hardware encoder wraps its output in an IVF container and emits the sequence
header only once, in the very first packet:

```text
packet 0 (dropped internally): [IVF headers] TD  SH  KEY_FRAME   <- SH appears only here
packet N (later key frame)   :               TD      KEY_FRAME   <- not decodable on its own
```

`JetsonAV1Compressor` therefore

1. strips the IVF file/frame headers (plain OBU streams are what `ffmpeg_image_transport`
   compatible decoders expect), and
2. caches the sequence header OBU from the first packet and re-inserts it right after the
   temporal delimiter of every subsequent key frame packet:

```text
packet N (later key frame)   :               TD  SH  KEY_FRAME   <- self-contained random access point
```

As a result, subscribers can join mid-stream and start decoding at any key frame, and every
packet still decodes to exactly one frame (only the ~10-20 byte sequence header is
duplicated, never a whole frame).

## Example Usage in ROS 2

The following code demonstrates how to leverage the compressor in your ROS 2 codebase:

```c++
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
