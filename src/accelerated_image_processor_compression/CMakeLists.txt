cmake_minimum_required(VERSION 3.16)
project(accelerated_image_processor_compression LANGUAGES CXX CUDA)

# Set CMAKE_MODULE_PATH to include the custom cmake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

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
else()
  set(JETSON_FOUND FALSE)
  message("Non-Jetson platform detected")
endif()

# --- nvJPEG available? ---
find_package(CUDAToolkit)
find_package(NVJPEG)
if(${NVJPEG_FOUND})
  message("NVJPEG found")
endif()

# --- LibJpegTurbo available? ---
find_package(LibJpegTurbo)
if(${LibJpegTurbo_FOUND})
  message("LibJpegTurbo found")
endif()

if(NOT JETSON_FOUND
   AND NOT NVJPEG_FOUND
   AND NOT LibJpegTurbo_FOUND)
  message(FATAL_ERROR "No JPEG encoder found")
endif()

# --- CUDA VPI available? ---
find_package(VPI)
if(${VPI_FOUND})
  message("VPI found")
endif()

# --- Jetson multimedia API available? ---
find_package(NVMMAPI)
if(${NVMMAPI_FOUND})
  message("Jetson multimedia API found")
endif()

add_library(${PROJECT_NAME} SHARED src/builder.cpp src/jpeg_compressor/cpu.cpp
                                   src/jpeg_compressor/nvjpeg.cpp)

target_compile_definitions(
  ${PROJECT_NAME}
  PUBLIC $<$<BOOL:${JETSON_FOUND}>:JETSON_AVAILABLE>
         $<$<BOOL:${NVJPEG_FOUND}>:NVJPEG_AVAILABLE>
         $<$<BOOL:${LibJpegTurbo_FOUND}>:TURBOJPEG_AVAILABLE>)
target_include_directories(
  ${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                         $<INSTALL_INTERFACE:include>)
target_include_directories(
  ${PROJECT_NAME}
  PRIVATE $<$<BOOL:${CUDAToolkit_FOUND}>:${CUDAToolkit_INCLUDE_DIRS}>
          $<$<BOOL:${NVJPEG_FOUND}>:${NVJPEG_INCLUDE_DIRS}>)
target_link_libraries(
  ${PROJECT_NAME}
  $<$<BOOL:${LibJpegTurbo_FOUND}>:${LibJpegTurbo_LIBRARIES}>
  $<$<TARGET_EXISTS:NVJPEG::nvjpeg>:NVJPEG::nvjpeg>
  $<$<TARGET_EXISTS:CUDA::nppc>:CUDA::nppc>
  $<$<TARGET_EXISTS:CUDA::nppicc>:CUDA::nppicc>
  $<$<TARGET_EXISTS:CUDA::nppidei>:CUDA::nppidei>
  $<$<TARGET_EXISTS:CUDA::nppig>:CUDA::nppig>
  $<$<TARGET_EXISTS:CUDA::nppisu>:CUDA::nppisu>)
ament_target_dependencies(${PROJECT_NAME} accelerated_image_processor_common)

if(JETSON_FOUND)
  target_include_directories(${PROJECT_NAME}
                             PRIVATE /usr/src/jetson_multimedia_api/include)
  add_library(
    ${PROJECT_NAME}_jetson SHARED
    # JPEG compression
    src/jpeg_compressor/jetson.cpp
    /usr/src/jetson_multimedia_api/samples/common/classes/NvBuffer.cpp
    /usr/src/jetson_multimedia_api/samples/common/classes/NvElement.cpp
    /usr/src/jetson_multimedia_api/samples/common/classes/NvElementProfiler.cpp
    /usr/src/jetson_multimedia_api/samples/common/classes/NvJpegEncoder.cpp
    /usr/src/jetson_multimedia_api/samples/common/classes/NvLogging.cpp
    # video compression
    src/video_compressor/jetson.cpp
    src/video_compressor/jetson_h264.cpp
    src/video_compressor/jetson_h265.cpp
    src/video_compressor/jetson_av1.cpp
    /usr/src/jetson_multimedia_api/samples/common/classes/NvVideoEncoder.cpp
    /usr/src/jetson_multimedia_api/samples/common/classes/NvV4l2Element.cpp
    /usr/src/jetson_multimedia_api/samples/common/classes/NvV4l2ElementPlane.cpp
    /usr/src/jetson_multimedia_api/samples/common/classes/NvBufSurface.cpp)
  target_compile_definitions(${PROJECT_NAME}_jetson PRIVATE JETSON_AVAILABLE)

  ament_target_dependencies(${PROJECT_NAME}_jetson
                            accelerated_image_processor_common)
  target_include_directories(
    ${PROJECT_NAME}_jetson
    PRIVATE $<$<BOOL:${CUDAToolkit_FOUND}>:${CUDAToolkit_INCLUDE_DIRS}>
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${SYS_ROOT}/usr/src/jetson_multimedia_api/include
            ${SYS_ROOT}/usr/src/jetson_multimedia_api/include/libjpeg-8b)
  find_library(
    JETSON_NVJPEG_LIB
    NAMES nvjpeg
    PATHS /usr/lib/aarch64-linux-gnu/tegra
    NO_DEFAULT_PATH)
  target_link_libraries(
    ${PROJECT_NAME}_jetson
    ${JETSON_NVJPEG_LIB}
    ${NVMMAPI_LIBRARIES}
    $<$<TARGET_EXISTS:CUDA::cudart>:CUDA::cudart>
    $<$<TARGET_EXISTS:CUDA::nppc>:CUDA::nppc>
    $<$<TARGET_EXISTS:CUDA::nppicc>:CUDA::nppicc>
    $<$<TARGET_EXISTS:CUDA::nppidei>:CUDA::nppidei>
    $<$<TARGET_EXISTS:CUDA::nppig>:CUDA::nppig>
    $<$<TARGET_EXISTS:CUDA::nppisu>:CUDA::nppisu>
    $<$<TARGET_EXISTS:vpi>:vpi>)

  set(JETSON_LIB_DIRS
      "/usr/lib/aarch64-linux-gnu/nvidia;/usr/lib/aarch64-linux-gnu/tegra")
  set_target_properties(
    ${PROJECT_NAME}_jetson PROPERTIES BUILD_RPATH "${JETSON_LIB_DIRS}"
                                      INSTALL_RPATH "${JETSON_LIB_DIRS}")
  target_link_options(${PROJECT_NAME}_jetson PRIVATE "-Wl,--disable-new-dtags")

  target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_jetson)
endif()

set_property(
  TARGET ${PROJECT_NAME}
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>)

if(JETSON_FOUND)
  install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_jetson
          EXPORT export_${PROJECT_NAME})
else()
  install(TARGETS ${PROJECT_NAME} EXPORT export_${PROJECT_NAME})
endif()

install(DIRECTORY include/${PROJECT_NAME} DESTINATION include)

if(BUILD_TESTING)
  set(test_files test/builder.cpp test/cpu_jpeg_compressor.cpp
                 test/jetson_jpeg_compressor.cpp test/nv_jpeg_compressor.cpp)

  foreach(test_file IN LISTS test_files)
    get_filename_component(test_file_name ${test_file} NAME)
    ament_auto_add_gtest(${test_file_name}_${PROJECT_NAME} ${test_file})
    target_link_libraries(${test_file_name}_${PROJECT_NAME} ${PROJECT_NAME})
  endforeach()

  # Treat some tests separately because they contains many test cases and cause
  # colcon test timeout if the tests are generated in the above way
  set(long_test_files
      test/jetson_video_compressor_h264.cpp
      test/jetson_video_compressor_h265.cpp
      test/jetson_video_compressor_av1.cpp)

  find_package(GTest)
  enable_testing()
  include(GoogleTest) # enable the GoogleTest integration

  foreach(test_file IN LISTS long_test_files)
    get_filename_component(test_file_name ${test_file} NAME)

    set(LONG_TEST_EXEC ${test_file_name}_${PROJECT_NAME})
    add_executable(${LONG_TEST_EXEC} ${test_file})
    target_include_directories(${LONG_TEST_EXEC}
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(${LONG_TEST_EXEC} ${PROJECT_NAME} GTest::GTest
                          GTest::Main)
    # divide a single test into separated cases
    gtest_discover_tests(${LONG_TEST_EXEC})
  endforeach()
endif()

ament_export_include_directories(include)
ament_export_targets(export_${PROJECT_NAME})
ament_export_dependencies(accelerated_image_processor_common)
ament_package()
