cmake_minimum_required(VERSION 3.14)
project(accelerated_image_processor_compression LANGUAGES CXX CUDA)

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

# --- Jetson available? ---
if(EXISTS "/etc/nv_tegra_release")
  set(JETSON_FOUND TRUE)
  message("Jetson platform detected")
  add_definitions(-DJETSON_AVAILABLE)
  set(TEGRA_LIBJPEG ${SYS_ROOT}/usr/lib/aarch64-linux-gnu/tegra/libjpeg.so)
else()
  set(JETSON_FOUND FALSE)
  message("Non-Jetson platform detected")
endif()

# --- nvJPEG available? ---
find_package(CUDA)
find_package(CUDAToolkit)
find_package(NVJPEG)
find_library(CULIBOS culibos ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
if(${NVJPEG_FOUND} AND CULIBOS)
  add_definitions(-DNVJPEG_AVAILABLE)
else()
  message("NVJPEG or its dependencies not found")
endif()

# --- LibJpegTurbo available? ---
find_package(LibJpegTurbo)
if(${LibJpegTurbo_FOUND})
  message("LibJpegTurbo found")
  add_definitions(-DTURBOJPEG_AVAILABLE)
endif()

if(NOT JETSON_FOUND
   AND NOT NVJPEG_FOUND
   AND NOT LibJpegTurbo_FOUND)
  message(FATAL_ERROR "No JPEG encoder found")
endif()

ament_auto_add_library(
  ${PROJECT_NAME}
  SHARED
  src/builder.cpp
  src/jpeg.cpp
  $<$<BOOL:${JETSON_FOUND}>:/usr/src/jetson_multimedia_api/samples/common/classes/NvBuffer.cpp>
  $<$<BOOL:${JETSON_FOUND}>:/usr/src/jetson_multimedia_api/samples/common/classes/NvElement.cpp>
  $<$<BOOL:${JETSON_FOUND}>:/usr/src/jetson_multimedia_api/samples/common/classes/NvElementProfiler.cpp>
  $<$<BOOL:${JETSON_FOUND}>:/usr/src/jetson_multimedia_api/samples/common/classes/NvJpegEncoder.cpp>
  $<$<BOOL:${JETSON_FOUND}>:/usr/src/jetson_multimedia_api/samples/common/classes/NvLogging.cpp>
)
target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
    $<$<BOOL:${CUDA_FOUND}>:${CUDA_INCLUDE_DIRS}>
    $<$<BOOL:${CUDAToolkit_FOUND}>:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>
    $<$<BOOL:${CUDAToolkit_FOUND}>:${CUDAToolkit_INCLUDE_DIRS}>
    $<$<BOOL:${JETSON_FOUND}>:${SYS_ROOT}/usr/src/jetson_multimedia_api/include>
    $<$<BOOL:${JETSON_FOUND}>:${SYS_ROOT}/usr/src/jetson_multimedia_api/include/libjpeg-8b>
)
target_link_directories(
  ${PROJECT_NAME} PUBLIC ${SYS_ROOT}/lib/aarch64-linux-gnu
  $<$<BOOL:${JETSON_FOUND}>:${SYS_ROOT}/usr/lib/aarch64-linux-gnu>
  $<$<BOOL:${JETSON_FOUND}>:${SYS_ROOT}/usr/lib/aarch64-linux-gnu/tegra>)
target_link_libraries(
  ${PROJECT_NAME}
  $<$<BOOL:${LibJpegTurbo_FOUND}>:${LibJpegTurbo_LIBRARIES}>
  $<$<BOOL:${NVJPEG_FOUND}>:${NVJPEG_LIBRARY}>
  $<$<BOOL:${JETSON_FOUND}>:nvjpeg>
  $<$<BOOL:${CUDAToolkit_FOUND}>:CUDA::cudart>
  $<$<TARGET_EXISTS:CUDA::nppicc>:CUDA::nppicc>
  $<$<TARGET_EXISTS:CUDA::nppidei>:CUDA::nppidei>
  $<$<TARGET_EXISTS:CUDA::nppisu>:CUDA::nppisu>)

ament_target_dependencies(${PROJECT_NAME} accelerated_image_processor_common)

set_property(TARGET ${PROJECT_NAME} PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                             $<INSTALL_INTERFACE:include>)
install(TARGETS ${PROJECT_NAME} EXPORT export_${PROJECT_NAME})
install(DIRECTORY include/${PROJECT_NAME}/ DESTINATION include)

if(BUILD_TESTING)
  set(test_files
      test/builder.cpp test/cpu_jpeg_compressor.cpp
      test/jetson_jpeg_compressor.cpp test/nvjpeg_jpeg_compressor.cpp)
  foreach(test_file IN LISTS test_files)
    get_filename_component(test_file_name ${test_file} NAME)
    ament_auto_add_gtest(${test_file_name}_${PROJECT_NAME} ${test_file})
    target_include_directories(
      ${test_file_name}_${PROJECT_NAME}
      PUBLIC
        $<$<BOOL:${CUDA_FOUND}>:${CUDA_INCLUDE_DIRS}>
        $<$<BOOL:${CUDAToolkit_FOUND}>:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>
        $<$<BOOL:${CUDAToolkit_FOUND}>:${CUDAToolkit_INCLUDE_DIRS}>
        $<$<BOOL:${JETSON_FOUND}>:${SYS_ROOT}/usr/src/jetson_multimedia_api/include>
        $<$<BOOL:${JETSON_FOUND}>:${SYS_ROOT}/usr/src/jetson_multimedia_api/include/libjpeg-8b>
    )
    target_link_directories(
      ${test_file_name}_${PROJECT_NAME}
      PUBLIC
      ${SYS_ROOT}/lib/aarch64-linux-gnu
      $<$<BOOL:${JETSON_FOUND}>:${SYS_ROOT}/usr/lib/aarch64-linux-gnu>
      $<$<BOOL:${JETSON_FOUND}>:${SYS_ROOT}/usr/lib/aarch64-linux-gnu/tegra>)
    target_link_libraries(
      ${test_file_name}_${PROJECT_NAME}
      ${PROJECT_NAME}
      $<$<BOOL:${LibJpegTurbo_FOUND}>:${LibJpegTurbo_LIBRARIES}>
      $<$<BOOL:${NVJPEG_FOUND}>:${NVJPEG_LIBRARY}>
      $<$<BOOL:${JETSON_FOUND}>:nvjpeg>
      $<$<BOOL:${CUDAToolkit_FOUND}>:CUDA::cudart>
      $<$<TARGET_EXISTS:CUDA::nppicc>:CUDA::nppicc>
      $<$<TARGET_EXISTS:CUDA::nppidei>:CUDA::nppidei>
      $<$<TARGET_EXISTS:CUDA::nppisu>:CUDA::nppisu>)
  endforeach()
endif()

ament_export_include_directories(include)
ament_export_targets(export_${PROJECT_NAME})
ament_export_dependencies(accelerated_image_processor_common)
ament_package()
