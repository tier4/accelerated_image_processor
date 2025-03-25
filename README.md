# accelerated_image_processor

A ROS2 package that provides GPU-accelerated image processing capabilities for efficient image rectification and compression. This package is designed to handle high-throughput image processing tasks using hardware acceleration when available.

## Features

- GPU-accelerated image rectification using:
  - NVIDIA Performance Primitives (NPP)
  - OpenCV CPU implementation
  - OpenCV CUDA implementation
- Hardware-accelerated JPEG compression using:
  - NVIDIA JPEG encoder (for Jetson platforms)
  - NVIDIA NVJPEG library
  - TurboJPEG (CPU fallback)
- Configurable processing pipeline
- Support for RGB8 and BGR8 image formats
- Task queue management for handling high-throughput scenarios
- ROS2 component-based architecture

## Dependencies

### Required
#### For GPU/HW acceleration
- CUDA Toolkit
- NVIDIA Performance Primitives (NPP)
- NVJPEG (for discrete GPU environment)
- Jetson Multimedia API (for Jetson platforms)
#### For CPU acceleration
- libturbojpeg

## Installation

1. Install the required dependencies:
```bash
sudo apt install ros-$ROS_DISTRO-cv-bridge ros-$ROS_DISTRO-image-geometry libturbojpeg0-dev
```

2. Clone this repository into your ROS2 workspace:
```bash
cd ~/ros2_ws/src
git clone https://github.com/tier4/accelerated_image_processor.git
```

3. Build the package:
```bash
cd ~/ros2_ws
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-up-to accelerated_image_processor
```

## Usage

The package provides a ROS2 component that can be loaded either as a standalone node or as part of a component container.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rect_impl` | string | "npp" | Rectification implementation to use ("npp", "opencv_cpu", or "opencv_gpu") |
| `use_opencv_map_init` | bool | false | Whether to use OpenCV for map initialization |
| `alpha` | double | 0.0 | Rectification alpha parameter |
| `jpeg_quality` | int | 60 | JPEG compression quality (0-100) |
| `do_rectify` | bool | true | Enable/disable image rectification |
| `max_task_queue_length` | int | 5 | Maximum number of images that can be queued for processing. A smaller value may cause dropped frames, while a larger value may lead to increased latency and higher memory usage. |

### Topics

#### Subscribed Topics
- `image_raw` (sensor_msgs/Image): Raw input image
- `camera_info` (sensor_msgs/CameraInfo): Camera calibration information

#### Published Topics
- `image_rect` (sensor_msgs/Image): Rectified image
- `image_rect/compressed` (sensor_msgs/CompressedImage): Compressed rectified image
- `image_raw/compressed` (sensor_msgs/CompressedImage): Compressed raw image

### Launch Examples

1. As a standalone node:
```bash
ros2 run accelerated_image_processor accelerated_image_processor_node
```

2. With custom parameters:
```bash
ros2 run accelerated_image_processor accelerated_image_processor_node --ros-args -p rect_impl:=npp -p jpeg_quality:=80 -p use_opencv_map_init:=true
```