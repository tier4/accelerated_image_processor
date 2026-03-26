# Accelerated Image Processor Python API

This package provides Python bindings of `accelerated_image_processor`.
It is intended to be a thin, performance-oriented wrapper around the C++ API.

Currently, the main public modules are:

- `accelerated_image_processor.common`: Common utilities and data structures.
- `accelerated_image_processor.compression`: Compression functionality.
- `accelerated_image_processor.decompression`: Decompression functionality.

## Installation

You can choose two types of installation depending on your demand.

### 1. Install with Python Package Manager

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

## Example of Usage

### 1. Load an image

```python
import numpy as np

from accelerated_image_processor.common import Image

# Load an image from a file path
image = Image.from_file("path/to/image")

# Load an image from a numpy array
height, width = 480, 640
image = Image.from_numpy(np.zeros((height, width, 3), dtype=np.uint8))
```

### 2. Compress an image

Currently, we support the following compression types: `JPEG`, `AV1`, `H264`, `H265`.

By invoking `create_compressor`, you can specify the compression type you want to use.

`compressor.register_postprocess(...)` enables you to register a postprocess function to handle the processed image.

```python
from accelerated_image_processor.compression import create_compressor, CompressionType

# Create a JPEG compressor
compressor = create_compressor(CompressionType.JPEG)

# Register a postprocess function
compressor.register_postprocess(lambda x: print(f"Processed image: (H, W) = ({x.height}, {x.width})"))

compressed_image = compressor.process(image)
```

Note that the video compressors compress images asynchronously, so we recommend handling the compressed image with a postprocess function.

```python
from accelerated_image_processor.compression import create_compressor, CompressionType

# Create a VIDEO compressor, you can choose from (AV1, H264, H265)
compressor = create_compressor(CompressionType.AV1)

# Register a postprocess function
compressor.register_postprocess(lambda x: print(f"Processed image: (H, W) = ({x.height}, {x.width})"))

# NOTE: compressed_image could be None even if the compression was successful
compressed_image = compressor.process(image)
```

### 3. Decompress an image

Currently, we support the following decompression types: `VIDEO`.

Video decompressor covers all video formats such as `AV1`, `H264`, `H265`.

By invoking `create_decompressor`, you can specify the decompression type you want to use.

As the video compression, we recommend handling the decompressed image with a postprocess function.

```python
from accelerated_image_processor.decompression import create_decompressor, DecompressionType

# Create a VIDEO decompressor
decompressor = create_decompressor(DecompressionType.VIDEO)

# Register a postprocess function
decompressor.register_postprocess(lambda x: print(f"Processed image: (H, W) = ({x.height}, {x.width})"))

decompressed_image = decompressor.process(compressed_image)
```

### 4. Update Parameters of the Processor

By calling `fetch_parameters(...)`, you can update the specific parameters of the processor.

```python
from accelerated_image_processor.common import fetch_parameters
from accelerated_image_processor.compression import create_compressor, CompressionType

compressor = create_compressor(CompressionType.JPEG)

print("Before update:", compressor.parameters)

config = {"quality": 10}
compressor = fetch_parameters(config, compressor)

print("After update:", compressor.parameters)
```

Then the following output will be printed:

```bash
Before update: {'quality': 90}
After update: {'quality': 10}
```
