# Accelerated Image Processor Python API

This package provides Python bindings of `accelerated_image_processor`.
It is intended to be a thin, performance-oriented wrapper around the C++ API.

Currently, the main public modules are:

- `accelerated_image_processor.common`: Common utilities and data structures.
- `accelerated_image_processor.compression`: Compression and decompression functionality.

## Installation

You can choose two types of installation depending on your demand.

### 1. Install with Python Package Manger

- `pip`

  ```bash
  pip install git+https://github.com/tier4/accelerated_image_processor
  ```

- `uv`

  ```bash
  uv add git+https://github.com/tier4/accelerated_image_processor
  ```

### 2. Build as ROS 2 Package

```bash
# 1. Clone and move workspace
git clone https://github.com/tier4/accelerated_image_processor && cd accelerated_image_processor

# 2. Install dependencies
rosdep update && rosdep install -y --from-paths src --ignore-src --rosdistro $ROS_DISTRO

# 3. Build and activate
colcon build --symlink-install --packages-up-to accelerated_image_processor_python
source install/setup.bash
```
