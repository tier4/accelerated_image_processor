cmake_minimum_required(VERSION 3.14)
project(accelerated_image_processor_decompression LANGUAGES CXX CUDA)

# Set CMAKE_MODULE_PATH to include the custom cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wunused-function)
endif()

if(NOT SKBUILD)
  find_package(ament_cmake_auto REQUIRED)
  ament_auto_find_build_dependencies()
endif()

find_package(CUDAToolkit REQUIRED)
find_package(FFNVCODEC 13 REQUIRED)
find_library(CUDA_nppicc_LIBRARY nppicc ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUDA_nppidei_LIBRARY nppidei
             ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUDA_nppisu_LIBRARY nppisu ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

add_library(
  ${PROJECT_NAME} SHARED src/builder.cpp src/video_decompressor/nvdec.cpp
                         src/video_decompressor/color_convert.cu)

target_include_directories(
  ${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                         $<INSTALL_INTERFACE:include>)

target_include_directories(${PROJECT_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} SYSTEM
                           PRIVATE ${FFNVCODEC_INCLUDE_DIRS})

target_link_libraries(
  ${PROJECT_NAME}
  PRIVATE CUDA::cudart
          CUDA::cuda_driver
          ${CUDA_nppicc_LIBRARY} # color conversion
          ${CUDA_nppisu_LIBRARY} # memory support functions
          ${CUDA_nppidei_LIBRARY} # data exchange and initialization functions
          ${CMAKE_DL_LIBS} # nv-codec-headers loads libnvcuvid at runtime
)

if(SKBUILD)
  target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC
      accelerated_image_processor_common::accelerated_image_processor_common)
else()
  ament_target_dependencies(${PROJECT_NAME} PUBLIC
                            accelerated_image_processor_common)
endif()

install(TARGETS ${PROJECT_NAME} EXPORT export_${PROJECT_NAME})
install(DIRECTORY include/${PROJECT_NAME} DESTINATION include)
install(FILES THIRD_PARTY_NOTICES.md DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING AND NOT SKBUILD)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_builder test/builder.cpp)
  target_link_libraries(test_builder ${PROJECT_NAME})

  # FFmpeg is used only to generate encoded packets for the decompressor tests.
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET libavcodec libavutil)
  pkg_check_modules(LIBAVFILTER REQUIRED IMPORTED_TARGET libavfilter)
  ament_add_gtest(test_nvdec_video_decompressor
                  test/nvdec_video_decompressor.cpp)
  target_link_libraries(test_nvdec_video_decompressor ${PROJECT_NAME}
                        PkgConfig::LIBAV PkgConfig::LIBAVFILTER)
endif()

if(NOT SKBUILD)
  ament_export_include_directories(include)
  ament_export_targets(export_${PROJECT_NAME})
  ament_export_dependencies(accelerated_image_processor_common)
  ament_package()
endif()
