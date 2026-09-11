# accelerated_image_processor_common

This package provides the common datatypes, parameter representations, and processor base classes shared by the other accelerated image processor packages.
It is a header-only interface library and has no ROS dependency.

## Headers

| Header          | Contents                                                                                                                    |
| --------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `datatype.hpp`  | `Image` and `CameraInfo` structs, and the `ImageEncoding`, `ImageFormat`, and `DistortionModel` enums                       |
| `parameter.hpp` | `ParameterKey`, `ParameterValue`, and `ParameterMap` used to expose processor parameters                                    |
| `processor.hpp` | `BaseProcessor`, the base class of every processor, which holds parameters and the callback invoked with a processed result |
| `helper.hpp`    | `CHECK_ERROR`, `CHECK_CUDA`, `CHECK_NPP`, and `CHECK_NVJPEG` error checking macros                                          |
