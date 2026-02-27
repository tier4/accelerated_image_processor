cmake_minimum_required(VERSION 3.14)
project(accelerated_image_processor_ros)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wunused-function)
endif()

find_package(ament_cmake_auto REQUIRED)
find_package(CUDAToolkit)
ament_auto_find_build_dependencies()

ament_auto_add_library(${PROJECT_NAME}_peripheral SHARED src/parameter.cpp
                       src/qos.cpp src/conversion.cpp)

target_include_directories(
  ${PROJECT_NAME}_peripheral
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>)

ament_auto_add_library(${PROJECT_NAME} SHARED src/imgproc_node.cpp)

target_include_directories(
  ${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                         $<INSTALL_INTERFACE:include>)

target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_peripheral)

ament_auto_add_library(${PROJECT_NAME}_decompress SHARED
                       src/decompress_node.cpp)

target_link_libraries(${PROJECT_NAME}_decompress ${PROJECT_NAME}_peripheral)

rclcpp_components_register_node(
  ${PROJECT_NAME} PLUGIN "accelerated_image_processor::ros::ImgProcNode"
  EXECUTABLE ${PROJECT_NAME}_imgproc_node)

rclcpp_components_register_node(
  ${PROJECT_NAME}_decompress PLUGIN
  "accelerated_image_processor::ros::DecompressNode" EXECUTABLE
  ${PROJECT_NAME}_decompress_node)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  set(test_files test/conversion.cpp test/parameter.cpp test/qos.cpp)
  foreach(test_file IN LISTS test_files)
    get_filename_component(test_file_name ${test_file} NAME)
    ament_add_gtest(${test_file_name}_${PROJECT_NAME} ${test_file})
    target_link_libraries(${test_file_name}_${PROJECT_NAME} ${PROJECT_NAME})
  endforeach()
endif()

ament_auto_package(INSTALL_TO_SHARE config launch)
