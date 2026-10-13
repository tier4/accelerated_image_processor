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

## Color Space Handling

The companion `accelerated_image_processor_compression` package encodes **full-range BT.709**
YCbCr (`nppiBGRToYCbCr420_709HDTV`). The decoder reproduces those colors as follows:

- **Matrix (BT.601 vs BT.709)**: taken from the stream's `matrix_coefficients`
  (`CUVIDEOFORMAT::video_signal_description`). `5`/`6` select BT.601; everything else, including
  `2` (unspecified), is treated as BT.709 — this project's convention.
- **Range flag is not used**: `video_full_range_flag` is intentionally _not_ read, because cuvid
  does **not** parse the AV1 `color_range` bit and always reports `0` (while `color_primaries` /
  `transfer_characteristics` / `matrix_coefficients` are parsed correctly).
- **BT.709 uses a custom kernel, not NPP.** For the BT.709 path the decoder applies the exact
  **ITU-R BT.709 full-range** `YCbCr -> RGB` matrix in a small CUDA kernel (`color_convert.cu`).

> **Why a custom kernel instead of NPP's fixed converters?** None of NPP's fixed `NV12ToBGR`
> variants correctly inverts the encoder's full-range `nppiBGRToYCbCr420_709HDTV`:
> `nppiNV12ToBGR_709HDTV` **desaturates** saturated colors by up to ~40 (e.g. pure red
> `255 -> 215`) — its forward/inverse are not an actual matched pair in NPP — while
> `nppiNV12ToBGR_709CSC` applies limited-range luma scaling that **crushes shadows and clips
> highlights** (gray `8 -> 0`, `16 -> 0`, `240 -> 255`) on this full-range content. Both were
> confirmed with a compression-free pure-NPP round-trip, so it is not codec loss. The custom
> kernel uses the standard ITU-R BT.709 full-range coefficients (derived from `Kr = 0.2126`,
> `Kb = 0.0722`), which is a well-defined standard — not an experimentally tuned matrix — and
> round-trips every swatch within ~3 per channel with no luma crush.

| Chroma       | Matrix | Conversion used                   | Notes                                                     |
| ------------ | ------ | --------------------------------- | --------------------------------------------------------- |
| 4:2:0 (NV12) | BT.709 | `nv12_to_bgr_bt709_full` kernel   | Standard ITU-R BT.709 full range; primary path            |
| 4:2:0 (NV12) | BT.601 | `nppiNV12ToBGR`                   | BT.601 fallback for external streams                      |
| 4:4:4        | BT.709 | `yuv444_to_bgr_bt709_full` kernel | Same matrix as NV12 (4:4:4 not exercised by the pipeline) |
| 4:4:4        | BT.601 | `nppiYCbCrToBGR`                  | BT.601 fallback                                           |

This design optimizes for the primary use case — decoding this package's own (full-range BT.709)
output. To support **external** streams that genuinely use limited range, read the range from the
stream (for H264/H265 cuvid parses `video_full_range_flag`; for AV1 parse the sequence header's
`color_range`, e.g. with `av1_sequence_header::parse_sequence_header`) and add a limited-range
variant of the kernel.

A standalone round-trip color verification tool lives in `external_tools/color_roundtrip_test.cpp`
(it encodes known swatches — including shadow/highlight grays to catch luma crushing — with NVENC
AV1, decodes them with NVDEC, and reports each channel's mean error). It needs a GPU and is built
on demand (`external_tools/CMakeLists.txt`), so it is kept out of the colcon build with a
`COLCON_IGNORE` marker rather than living under `test/` with the unit tests.

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
