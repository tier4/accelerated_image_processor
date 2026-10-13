cmake_minimum_required(VERSION 3.14)
project(color_roundtrip_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The compression export references CUDA targets in its public link interface.
find_package(CUDAToolkit REQUIRED)
find_package(accelerated_image_processor_common REQUIRED)
find_package(accelerated_image_processor_compression REQUIRED)
find_package(accelerated_image_processor_decompression REQUIRED)

add_executable(color_roundtrip_test color_roundtrip_test.cpp)
target_compile_options(color_roundtrip_test PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(
  color_roundtrip_test
  PRIVATE
    accelerated_image_processor_compression::accelerated_image_processor_compression
    accelerated_image_processor_decompression::accelerated_image_processor_decompression
)
