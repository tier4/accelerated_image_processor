cmake_minimum_required(VERSION 3.14)
project(accelerated_image_processor_pipeline LANGUAGES CXX CUDA)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wunused-function)
endif()

find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV REQUIRED)
ament_auto_find_build_dependencies()

# --- NPP available? ---
find_package(CUDAToolkit)
find_library(CUDA_nppicc_LIBRARY nppicc ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUDA_nppidei_LIBRARY nppidei
             ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUDA_nppig_LIBRARY nppig ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUDA_nppisu_LIBRARY nppisu ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
if(CUDA_nppicc_LIBRARY
   AND CUDA_nppidei_LIBRARY
   AND CUDA_nppig_LIBRARY
   AND CUDA_nppisu_LIBRARY)
  set(NPP_FOUND TRUE)
  add_definitions(-DNPP_AVAILABLE)
else()
  set(NPP_FOUND FALSE)
endif()

add_library(
  ${PROJECT_NAME} SHARED
  src/builder.cpp src/rectifier/cpu.cpp src/rectifier/npp.cpp
  src/rectifier/opencv_cuda.cpp src/rectifier/utility.cpp)

target_compile_definitions(
  ${PROJECT_NAME}
  PUBLIC $<$<BOOL:${NPP_FOUND}>:NPP_AVAILABLE>
         $<$<BOOL:${OpenCV_CUDA_VERSION}>:OPENCV_CUDA_AVAILABLE>)
target_include_directories(
  ${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                         $<INSTALL_INTERFACE:include>)
target_include_directories(
  ${PROJECT_NAME}
  PRIVATE $<$<BOOL:${CUDAToolkit_FOUND}>:${CUDAToolkit_INCLUDE_DIRS}>)
target_link_libraries(
  ${PROJECT_NAME}
  ${OpenCV_LIBRARIES}
  $<$<BOOL:${NPP_FOUND}>:${CUDA_nppidei_LIBRARY}>
  $<$<BOOL:${NPP_FOUND}>:${CUDA_nppig_LIBRARY}>
  $<$<BOOL:${NPP_FOUND}>:${CUDA_nppicc_LIBRARY}>
  $<$<BOOL:${NPP_FOUND}>:${CUDA_nppisu_LIBRARY}>
  $<$<BOOL:${CUDAToolkit_FOUND}>:CUDA::cudart>)
ament_target_dependencies(${PROJECT_NAME} accelerated_image_processor_common)

set_property(TARGET ${PROJECT_NAME} PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                             $<INSTALL_INTERFACE:include>)
install(TARGETS ${PROJECT_NAME} EXPORT export_${PROJECT_NAME})
install(DIRECTORY include/${PROJECT_NAME}/ DESTINATION include)

if(BUILD_TESTING)
  set(test_files test/builder.cpp test/cpu_rectifier.cpp test/npp_rectifier.cpp
                 test/opencv_cuda_rectifier.cpp)
  foreach(test_file IN LISTS test_files)
    get_filename_component(test_file_name ${test_file} NAME)
    ament_auto_add_gtest(${test_file_name}_${PROJECT_NAME} ${test_file})
    target_link_libraries(${test_file_name}_${PROJECT_NAME} ${PROJECT_NAME})
  endforeach()
endif()

ament_export_include_directories(include)
ament_export_targets(export_${PROJECT_NAME})
ament_export_dependencies(accelerated_image_processor_common)
ament_package()
