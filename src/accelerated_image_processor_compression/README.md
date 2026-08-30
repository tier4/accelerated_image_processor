# accelerated_image_processor_compression

This package provides compression functionalities for images using various algorithms.
It also includes support for hardware acceleration on NVIDIA Jetson devices using the Jetson Multimedia API,
and on x86 hosts with a discrete NVIDIA GPU using NVENC (NvEncodeAPI).

## Compressor Supports

| Compressor             | Format | Backend                                                                                  | Device |
| ---------------------- | ------ | ---------------------------------------------------------------------------------------- | ------ |
| `JetsonJPEGCompressor` | `JPEG` | [jetsonJPEG](https://docs.nvidia.com/jetson/l4t-multimedia/classNvJPEGEncoder.html)      | Jetson |
| `NvJPEGCompressor`     | `JPEG` | [nvJPEG](https://developer.nvidia.com/nvjpeg)                                            | GPU    |
| `CpuJPEGCompressor`    | `JPEG` | [TurboJPEG](https://github.com/libjpeg-turbo/libjpeg-turbo)                              | CPU    |
| `JetsonH264Compressor` | `H264` | [NvVideoEncoder](https://docs.nvidia.com/jetson/l4t-multimedia/classNvVideoEncoder.html) | Jetson |
| `JetsonH265Compressor` | `H265` | [NvVideoEncoder](https://docs.nvidia.com/jetson/l4t-multimedia/classNvVideoEncoder.html) | Jetson |
| `JetsonAV1Compressor`  | `AV1`  | [NvVideoEncoder](https://docs.nvidia.com/jetson/l4t-multimedia/classNvVideoEncoder.html) | Jetson |
| `NvencAV1Compressor`   | `AV1`  | [NVENC (NvEncodeAPI)](https://developer.nvidia.com/nvidia-video-codec-sdk)               | GPU    |

## AV1 bitstream handling

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

### What `JetsonAV1Compressor` does

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

### What `NvencAV1Compressor` does

NVENC emits a plain OBU stream where each packet is already a single temporal unit, and it inserts
the sequence header into every IDR frame on its own. Therefore neither header stripping nor
sequence header insertion is needed:

```text
packet N (key frame) : TD  SH  KEY_FRAME   <- emitted by the encoder as is
```

The sequence header is, however, tied to **IDR** frames. If the IDR period were longer than the
GOP length, NVENC would emit the intra frames in between as AV1 `INTRA_ONLY` frames, which carry
no sequence header. Such a frame must never be advertised as a key frame: unlike a key frame, an
`INTRA_ONLY` frame does not reset the reference frames, so a decoder cannot start decoding at it
even if a sequence header were spliced in front of it.

`NvencAV1Compressor` therefore drives the GOP length and the IDR period with the same value, the
smaller of `i_frame_interval` and `idr_frame_interval`. Every packet flagged as a key frame is
then an actual AV1 key frame accompanied by a sequence header, that is, a random access point.
AV1 needs no equivalent of the H264/H265 "I frame that is not an IDR frame" anyway, because its
key frame already resets the reference frames.

## Video compression on a non-Jetson platform (`NvencAV1Compressor`)

`NvencAV1Compressor` is the counterpart of `JetsonAV1Compressor` for an x86 host with a discrete
NVIDIA GPU. While Jetson devices expose their hardware encoder through the V4L2 based Jetson
Multimedia API, a discrete GPU exposes it through **NvEncodeAPI (NVENC)**, whose headers are
provided by [nv-codec-headers](https://github.com/FFmpeg/nv-codec-headers) and whose
implementation lives in `libnvidia-encode.so.1` shipped with the NVIDIA display driver.

`accelerated_image_processor_compression::create_compressor("av1")` picks the backend
automatically: `JetsonAV1Compressor` on a Jetson device, `NvencAV1Compressor` otherwise.

### Requirements

| Item             | Requirement                                                                                   |
| ---------------- | --------------------------------------------------------------------------------------------- |
| GPU              | NVENC engine that supports AV1 encoding (Ada Lovelace generation or newer)                    |
| Display driver   | Recent enough for the NvEncodeAPI version the headers declare (`13.0` needs `570.0` or later) |
| nv-codec-headers | Headers only. Resolved at build time (see below)                                              |
| CUDA Toolkit     | The color conversion from RGB/BGR to YUV runs on NPP                                          |

The NVENC backend is enabled only when the CUDA Toolkit and nv-codec-headers are both found on a
non-Jetson platform, in which case CMake reports
`nv-codec-headers found (NvEncodeAPI <version>): NVENC video compression enabled`.
`libnvidia-encode.so.1` is loaded lazily with `dlopen()`, so the built library stays loadable on a
host that has no NVIDIA driver installed. When the driver or the GPU turns out to be unusable, the
compressor reports it through the initialization error instead of crashing.

### Resolving nv-codec-headers

`cmake/FindFFNVCODEC.cmake` searches the headers in the following order:

1. `FFNVCODEC_ROOT_DIR`, which defaults to a nv-codec-headers checkout placed next to this
   repository (see below)
2. A system installation, found via `pkg-config ffnvcodec` or the standard include directories
3. A download from GitHub via `FetchContent`

The default location is the sibling of this repository, that is the layout `vcs import` produces
for a workspace build:

```text
<workspace>/src/accelerated_image_processor   <- this repository
<workspace>/src/nv-codec-headers              <- the headers
```

To place the headers there, either clone them next to this repository, or add the entry to a
`.repos` file that `vcs import src` consumes:

```yaml
repositories:
  nv-codec-headers:
    type: git
    url: https://github.com/FFmpeg/nv-codec-headers.git
    version: n13.0.19.1
```

The following CMake options control the behavior:

| Option                     | Default                                         | Description                                                   |
| -------------------------- | ----------------------------------------------- | ------------------------------------------------------------- |
| `FFNVCODEC_ROOT_DIR`       | `<sibling of this repository>/nv-codec-headers` | Folder that contains `include/ffnvcodec` of nv-codec-headers  |
| `FFNVCODEC_ALLOW_DOWNLOAD` | `ON`                                            | Whether the headers may be downloaded when they are not found |
| `FFNVCODEC_DOWNLOAD_TAG`   | `n13.0.19.1`                                    | git tag to be downloaded                                      |

NvEncodeAPI is **not** forward compatible: the display driver has to be equal to or newer than the
SDK the headers come from. Hence a conservative tag is pinned by default, and a newer SDK has to be
requested explicitly, for instance with
`colcon build --cmake-args -DFFNVCODEC_DOWNLOAD_TAG=n13.1.15.0`.

### Parameters

The parameter keys are kept identical to the Jetson ones wherever the meaning carries over, so that
a single parameter file can drive both backends.

| Parameter                | Type     | Default             | Description                                                                                                                                                                                     |
| ------------------------ | -------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `compression_type`       | `string` | `lossy`             | `lossy` only. NVENC does not expose lossless encoding for AV1, hence `lossless` is rejected with an error message                                                                               |
| `idr_frame_interval`     | `int`    | `10`                | Key frame interval in frames. The smaller of this and `i_frame_interval` drives both `NV_ENC_CONFIG_AV1::idrPeriod` and `NV_ENC_CONFIG::gopLength` (see [above](#what-nvencav1compressor-does)) |
| `i_frame_interval`       | `int`    | `10`                | Key frame interval in frames. Interchangeable with `idr_frame_interval` on this backend, because every AV1 key frame is emitted as an IDR frame                                                 |
| `frame_rate_numerator`   | `int`    | `10`                | Numerator of the frame rate, in frames                                                                                                                                                          |
| `frame_rate_denominator` | `int`    | `1`                 | Denominator of the frame rate, in seconds                                                                                                                                                       |
| `buffer_length`          | `int`    | `4`                 | Number of input/bitstream buffer pairs the encoder cycles through                                                                                                                               |
| `target_bits_per_pixel`  | `double` | `0.1`               | Target bitrate per pixel. The target bitrate is `width * height * frame_rate * this value`, with a 1.2x peak                                                                                    |
| `hw_preset_type`         | `string` | `medium`            | Encode preset. `p1` (fastest) to `p7` (best quality), or the Jetson names as aliases (see below)                                                                                                |
| `tuning_info`            | `string` | `ultra_low_latency` | `high_quality`, `ultra_high_quality`, `low_latency` or `ultra_low_latency`                                                                                                                      |
| `gpu_id`                 | `int`    | `0`                 | Index of the CUDA device that runs the encoding                                                                                                                                                 |
| `av1.enable_tile`        | `bool`   | `true`              | Enable tiling division, which lets the hardware encode the tiles in parallel                                                                                                                    |
| `av1.log2_num_tile_row`  | `int`    | `1`                 | Number of tile rows in log2, that is, `1` means 2 rows                                                                                                                                          |
| `av1.log2_num_tile_col`  | `int`    | `1`                 | Number of tile columns in log2, that is, `1` means 2 columns                                                                                                                                    |

`hw_preset_type` accepts the Jetson hardware preset names so that a parameter file written for the
Jetson backend keeps working. They are mapped to the closest NVENC preset:

| Jetson name | NVENC preset             |
| ----------- | ------------------------ |
| `ultrafast` | `P1`                     |
| `fast`      | `P3`                     |
| `medium`    | `P4`                     |
| `slow`      | `P6`                     |
| `disable`   | `P4` (the default grade) |

The following Jetson parameters have no NVENC counterpart and therefore are **not** exposed:
`use_max_performance` (a Jetson clock boost), `av1.enable_ssim_rdo` and `av1.enable_cdf_update`.
Unknown parameters in a parameter file are simply ignored by
`accelerated_image_processor::ros::fetch_parameters()`, so sharing a file between the backends is
harmless.

### Differences from the Jetson backend

| Aspect             | `JetsonAV1Compressor`                                             | `NvencAV1Compressor`                                              |
| ------------------ | ----------------------------------------------------------------- | ----------------------------------------------------------------- |
| Result delivery    | Asynchronous: the capture plane dequeue thread emits packets      | Synchronous: `process()` returns the packet of the frame just fed |
| Color conversion   | VPI (`NVBUF_COLOR_FORMAT_NV12_ER`, full range)                    | NPP (`nppi*ToYCbCr420_JPEG_*`, full range) into planar YUV 4:2:0  |
| Lossless encoding  | Supported                                                         | Not supported for AV1 (rejected during the parameter validation)  |
| Container overhead | IVF headers have to be stripped from every packet                 | None: NVENC emits a plain OBU stream                              |
| Sequence header    | Cached from the first packet and re-inserted into every key frame | Emitted by the encoder for every key frame                        |

Note that the registered postprocess function is invoked in both cases, hence the user code can be
shared between the backends.

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
