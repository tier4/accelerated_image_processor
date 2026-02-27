cmake_minimum_required(VERSION 3.14)
project(accelerated_image_processor_decompression LANGUAGES CXX CUDA)

# Set CMAKE_MODULE_PATH to include the custom cmake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wunused-function)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# find_package(CUDA)
find_package(CUDAToolkit)
find_library(CUDA_nppicc_LIBRARY nppicc ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUDA_nppidei_LIBRARY nppidei
             ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUDA_nppisu_LIBRARY nppisu ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

# --------------------------------------------------------------------------------
# Find FFMpeg shared libraries manually ref:
# https://github.com/ros-misc-utilities/ffmpeg_encoder_decoder/blob/master/CMakeLists.txt
# --------------------------------------------------------------------------------
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET libavcodec libavutil)
# --------------------------------------------------------------------------------

add_library(${PROJECT_NAME} SHARED src/builder.cpp
                                   src/video_decompressor/ffmpeg.cpp)

target_include_directories(
  ${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                         $<INSTALL_INTERFACE:include>)

target_include_directories(
  ${PROJECT_NAME}
  PRIVATE $<$<BOOL:${CUDAToolkit_FOUND}>:${CUDAToolkit_INCLUDE_DIRS}>)

target_link_libraries(
  ${PROJECT_NAME}
  PRIVATE PkgConfig::LIBAV
          CUDA::cudart
          CUDA::cuda_driver
          ${CUDA_nppicc_LIBRARY} # color conversion
          ${CUDA_nppisu_LIBRARY} # memory support functions
          ${CUDA_nppidei_LIBRARY} # data exchange and initialization functions
)

ament_target_dependencies(${PROJECT_NAME} PUBLIC
                          accelerated_image_processor_common)

install(TARGETS ${PROJECT_NAME} EXPORT export_${PROJECT_NAME})
install(DIRECTORY include/${PROJECT_NAME} DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_builder test/builder.cpp)
  target_link_libraries(test_builder ${PROJECT_NAME})

  pkg_check_modules(LIBAVFILTER REQUIRED IMPORTED_TARGET libavfilter)
  ament_add_gtest(test_ffmpeg_video_decompressor
                  test/ffmpeg_video_decompressor.cpp)
  target_link_libraries(test_ffmpeg_video_decompressor ${PROJECT_NAME}
                        PkgConfig::LIBAV PkgConfig::LIBAVFILTER)
endif()

ament_export_include_directories(include)
ament_export_targets(export_${PROJECT_NAME})
ament_export_dependencies(accelerated_image_processor_common)
ament_package()
