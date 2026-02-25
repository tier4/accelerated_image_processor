cmake_minimum_required(VERSION 3.16)
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

file(GLOB_RECURSE SOURCE_FILES src/*.cpp)
ament_auto_add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})

target_include_directories(
  ${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                         $<INSTALL_INTERFACE:include>)

rclcpp_components_register_node(
  ${PROJECT_NAME} PLUGIN "accelerated_image_processor::ros::ImgProcNode"
  EXECUTABLE ${PROJECT_NAME}_imgproc_node)

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
