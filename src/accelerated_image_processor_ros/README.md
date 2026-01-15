# accelerated_image_processor_ros

This package provides ROS-related nodes or plugins using the other accelerated image processor packages.

## `ImgProcNode`

`ImgProcNode` is a comprehensive image processing node that performs various image processing tasks such as image compression and rectification in a single node.

### Launch Standalone Node

```shell
ros2 launch accelerated_image_processor_ros imgproc.launch.xml
```

### I/O Topics

#### Inputs

| Name          | Type                     | Description                                                |
| ------------- | ------------------------ | ---------------------------------------------------------- |
| `image_raw`   | `sensor_msgs/Image`      | Input raw image                                            |
| `camera_info` | `sensor_msgs/CameraInfo` | Camera info (Required only if `rectifier.do_rectify=true`) |

#### Outputs

| Name                     | Type                          | Description                                                                |
| ------------------------ | ----------------------------- | -------------------------------------------------------------------------- |
| `image_raw/compressed`   | `sensor_msgs/CompressedImage` | Compressed image                                                           |
| `image_rect`             | `sensor_msgs/Image`           | Rectified image (Published only if `rectifier.do_rectify=true`)            |
| `image_rect/camera_info` | `sensor_msgs/CameraInfo`      | Rectified camera info (Published only if `rectifier.do_rectify=true`)      |
| `image_rect/compressed`  | `sensor_msgs/CompressedImage` | Compressed rectified image (Published only if `rectifier.do_rectify=true`) |

### Parameters

| Name                   | Type     | Default | Description                                    |
| ---------------------- | -------- | ------- | ---------------------------------------------- |
| `max_task_length`      | `int`    | `5`     | Maximum number of tasks to process in parallel |
| `compressor.type`      | `string` | `jpeg`  | Compression type [jpeg, video]                 |
| `compressor.quality`   | `int`    | `90`    | Compression quality                            |
| `rectifier.do_rectify` | `bool`   | `true`  | Whether to rectify the image                   |
| `rectifier.alpha`      | `double` | `0.0`   | Rectification alpha parameter                  |
