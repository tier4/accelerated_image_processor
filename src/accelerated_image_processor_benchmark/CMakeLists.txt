cmake_minimum_required(VERSION 3.8)
project(accelerated_image_processor_benchmark)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(accelerated_image_processor_common REQUIRED)
find_package(accelerated_image_processor_compression REQUIRED)

add_executable(compression_encode_bench src/compression_encode_bench.cpp)
ament_target_dependencies(
  compression_encode_bench accelerated_image_processor_common
  accelerated_image_processor_compression)
install(TARGETS compression_encode_bench DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights comment the
  # line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo) comment the line
  # when this package is in a git repo and when a copyright and license is added
  # to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
