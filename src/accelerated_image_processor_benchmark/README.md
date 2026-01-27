# accelerated_image_processor_benchmark

## Installation

```bash
colcon build --symlink-install -DCMAKE_BUILD_TYPE=Release --packages-up-to accelerated_image_processor_benchmark

source install/setup.bash
```

## Benchmark for Compression

To get help, run `compression -h`.

### Benchmark with Synthetic Image Data

```bash
# [] is optional arguments
compression <CONFIG_PATH> [--height <HEIGHT> --width <WIDTH> ...]
```

### Benchmark with ROSBag

```bash
# [] is optional arguments
compression <CONFIG_PATH> --bag <BAG_DIR> --topic <TOPIC_NAME> [--storage-id <sqlite3, mcap> ...]
```
