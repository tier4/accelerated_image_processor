# accelerated_image_processor

`accelerated_image_processor` is a set of C++ and Python libraries for accelerated image processing.
It provides common image data structures, image/video compression, video decompression, rectification pipelines, ROS 2 nodes, and benchmark tools.

> [!NOTE] > `src/accelerated_image_processor` is a legacy implementation and is intentionally not described here.
> The current implementation is split into the packages listed below.

## Packages

| Package                                                                                                  | Role                                                     | ROS dependency |
| -------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- | -------------- |
| [`accelerated_image_processor_common`](./src/accelerated_image_processor_common/README.md)               | Common datatypes, parameters, and processor base classes | No             |
| [`accelerated_image_processor_compression`](./src/accelerated_image_processor_compression/README.md)     | JPEG/video compression processors                        | No             |
| [`accelerated_image_processor_decompression`](./src/accelerated_image_processor_decompression/README.md) | CUDA-accelerated FFmpeg video decompression              | No             |
| [`accelerated_image_processor_pipeline`](./src/accelerated_image_processor_pipeline/README.md)           | Rectification processors                                 | No             |
| [`accelerated_image_processor_python`](./src/accelerated_image_processor_python/README.md)               | Python bindings for common/compression/decompression     | No             |
| [`accelerated_image_processor_ros`](./src/accelerated_image_processor_ros/README.md)                     | ROS 2 components/nodes and ROS message conversions       | Yes            |
| [`accelerated_image_processor_benchmark`](./src/accelerated_image_processor_benchmark/README.md)         | Benchmark CLI/library                                    | Yes            |

## Supported processors

### Compression

| Processor              | Format | Backend                                | Device/platform  |
| ---------------------- | ------ | -------------------------------------- | ---------------- |
| `CpuJPEGCompressor`    | `JPEG` | TurboJPEG                              | CPU              |
| `NvJPEGCompressor`     | `JPEG` | nvJPEG                                 | CUDA-capable GPU |
| `JetsonJPEGCompressor` | `JPEG` | Jetson Multimedia API                  | NVIDIA Jetson    |
| `JetsonH264Compressor` | `H264` | Jetson Multimedia API / NvVideoEncoder | NVIDIA Jetson    |
| `JetsonH265Compressor` | `H265` | Jetson Multimedia API / NvVideoEncoder | NVIDIA Jetson    |
| `JetsonAV1Compressor`  | `AV1`  | Jetson Multimedia API / NvVideoEncoder | NVIDIA Jetson    |

JPEG backend selection is automatic in priority order: Jetson, nvJPEG, then TurboJPEG.
Video compression is currently Jetson-only.

### Decompression

| Processor                 | Input formats         | Backend           | Device/platform  |
| ------------------------- | --------------------- | ----------------- | ---------------- |
| `FfmpegVideoDecompressor` | `H264`, `H265`, `AV1` | FFmpeg + CUDA/NPP | CUDA-capable GPU |

### Pipeline

| Processor             | Task          | Backend                             | Device/platform  |
| --------------------- | ------------- | ----------------------------------- | ---------------- |
| `NppRectifier`        | Rectification | NVIDIA Performance Primitives (NPP) | CUDA-capable GPU |
| `OpenCvCudaRectifier` | Rectification | OpenCV CUDA                         | CUDA-capable GPU |
| `CpuRectifier`        | Rectification | OpenCV                              | CPU              |

Rectifier backend selection is automatic in priority order: NPP, OpenCV CUDA, then CPU.

## Installation

### ROS 2 workspace build

Clone into a ROS 2 workspace and build only the current packages.

```bash
git clone https://github.com/tier4/accelerated_image_processor.git
cd accelerated_image_processor

rosdep update && rosdep install -y --from-paths src --ignore-src --rosdistro ${ROS_DISTRO}

colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

### Python package in a non-ROS CUDA environment

This repository can be installed as a Python package without ROS 2.
System packages are still required for native extensions.

Example for Ubuntu 22.04 + CUDA environment:

```bash
sudo apt update && sudo apt install -y \
  build-essential \
  cmake \
  git \
  libavcodec-dev \
  libavutil-dev \
  libboost-python-dev \
  libopencv-dev \
  libturbojpeg0-dev \
  ninja-build \
  pkg-config \
  python3-dev \
  python3-pip
```

Install with `uv`:

```bash
uv add git+https://github.com/tier4/accelerated_image_processor.git
```

Or install with `pip`:

```bash
pip install git+https://github.com/tier4/accelerated_image_processor.git
```
