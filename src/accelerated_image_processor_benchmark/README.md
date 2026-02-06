# accelerated_image_processor_benchmark

## Installation

```bash
colcon build --symlink-install -DCMAKE_BUILD_TYPE=Release --packages-up-to accelerated_image_processor_benchmark

source install/setup.bash
```

## Run Benchmarks

To get help, run `accbench -h`.

### Compression

To get help, run `accbench compression -h`.

#### Synthetic Image Data

```bash
# [] is optional arguments
accbench compression <CONFIG_PATH> [--height <HEIGHT> --width <WIDTH> ...]
```

#### RosBag

```bash
# [] is optional arguments
accbench compression <CONFIG_PATH> --bag <BAG_DIR> --topic <TOPIC_NAME> [--storage-id <sqlite3, mcap> ...]
```
