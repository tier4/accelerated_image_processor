cmake_minimum_required(VERSION 3.8)
project(accelerated_image_processor_pipeline)

find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV REQUIRED)
ament_auto_find_build_dependencies()

# --- NPP available? ---
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

# --- OpenCV CUDA available? ---
if(OpenCV_CUDA_VERSION)
  # Found OpenCV with CUDA support
  message("OpenCV CUDA version: ${OpenCV_CUDA_VERSION}")
  add_definitions(-DOPENCV_CUDA_AVAILABLE)
endif()

ament_auto_add_library(${PROJECT_NAME} SHARED src/builder.cpp src/rectifier.cpp)

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include> ${CUDA_INCLUDE_DIRS}
         ${OpenCV_INCLUDE_DIRS} $<$<BOOL:${NPP_FOUND}>:${CUDA_NPP_INCLUDES}>)
target_link_libraries(
  ${PROJECT_NAME}
  ${OpenCV_LIBRARIES}
  $<$<BOOL:${NPP_FOUND}>:${CUDA_nppidei_LIBRARY}>
  $<$<BOOL:${NPP_FOUND}>:${CUDA_nppig_LIBRARY}>
  $<$<BOOL:${NPP_FOUND}>:${CUDA_nppicc_LIBRARY}>
  $<$<BOOL:${NPP_FOUND}>:${CUDA_nppisu_LIBRARY}>)

ament_target_dependencies(${PROJECT_NAME} accelerated_image_processor_common)

set_property(TARGET ${PROJECT_NAME} PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                             $<INSTALL_INTERFACE:include>)
install(TARGETS ${PROJECT_NAME} EXPORT export_${PROJECT_NAME})
install(DIRECTORY include/${PROJECT_NAME}/ DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(include)
ament_export_targets(export_${PROJECT_NAME})
ament_export_dependencies(accelerated_image_processor_common)
ament_package()
