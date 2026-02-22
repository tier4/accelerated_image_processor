cmake_minimum_required(VERSION 3.14)
project(accelerated_image_processor_python)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(python_cmake_module REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(Boost REQUIRED COMPONENTS python)
find_package(accelerated_image_processor_common REQUIRED)

set(CMAKE_SHARED_LIBRARY_PREFIX "")
set(PYTHON_PACKAGE_NAME accelerated_image_processor)
ament_python_install_package(${PYTHON_PACKAGE_NAME})

set(PYTHON_MODULE_LIST) # container to store modules
set(MODULE_NAMES datatype)
foreach(MODULE_NAME ${MODULE_NAMES})
  set(PYTHON_MODULE_NAME ${PROJECT_NAME}_${MODULE_NAME})
  add_library(${PYTHON_MODULE_NAME} MODULE "src/${MODULE_NAME}.cpp")
  target_include_directories(
    ${PYTHON_MODULE_NAME}
    PUBLIC ${BoostPython_INCLUDE_DIRS}
           ${accelerated_image_processor_common_INCLUDE_DIRS})
  target_link_libraries(
    ${PYTHON_MODULE_NAME} Boost::python Python3::Python
    accelerated_image_processor_common::accelerated_image_processor_common)
  set_target_properties(${PYTHON_MODULE_NAME} PROPERTIES PREFIX "" SUFFIX ".so")
  list(APPEND PYTHON_MODULE_LIST ${PYTHON_MODULE_NAME})
endforeach()
install(TARGETS ${PYTHON_MODULE_LIST}
        LIBRARY DESTINATION "${PYTHON_INSTALL_DIR}/${PYTHON_PACKAGE_NAME}")

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  set(test_files test/test_datatype.py)
  foreach(test_file ${test_files})
    get_filename_component(test_name ${test_file} NAME_WE)
    ament_add_pytest_test(
      ${test_name}
      ${test_file}
      APPEND_ENV
      AMENT_PREFIX_PATH=${ament_index_build_path}
      PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}
      TIMEOUT
      120
      WERROR
      ON)
  endforeach()
endif()

ament_package()
