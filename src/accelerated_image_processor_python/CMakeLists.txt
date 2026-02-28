cmake_minimum_required(VERSION 3.16)
project(accelerated_image_processor_python)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(_AIP_RPATH "")
if(UNIX AND NOT APPLE)
  set(_AIP_RPATH "\$ORIGIN")
endif()

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

# find accelerated_image_processor packages
find_package(accelerated_image_processor_common REQUIRED)
find_package(accelerated_image_processor_compression REQUIRED)

# define python package name
set(PYTHON_PACKAGE_NAME accelerated_image_processor)
ament_python_install_package(${PYTHON_PACKAGE_NAME})

if(SKBUILD)
  # When scikit-build-core drives the build it defines the SKBUILD variable, so
  # always force PYTHON_INSTALL_DIR to the in-tree package layout in that
  # scenario to avoid polluting the outer environment.
  set(PYTHON_INSTALL_DIR
      "."
      CACHE PATH "Python install base directory" FORCE)

  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${PYTHON_PACKAGE_NAME}"
          DESTINATION "${PYTHON_INSTALL_DIR}")

  # list of shared libraries to be installed into target python package
  # directory
  set(_aip_embedded_targets accelerated_image_processor_compression)

  set(_aip_python_lib_dir "${PYTHON_INSTALL_DIR}/${PYTHON_PACKAGE_NAME}")
  foreach(_aip_target IN LISTS _aip_embedded_targets)
    if(TARGET ${_aip_target})
      set_target_properties(${_aip_target} PROPERTIES BUILD_WITH_INSTALL_RPATH
                                                      TRUE)

      if(_AIP_RPATH)
        set_target_properties(${_aip_target} PROPERTIES INSTALL_RPATH
                                                        "${_AIP_RPATH}")
      endif()

      install(
        TARGETS ${_aip_target}
        LIBRARY DESTINATION "${_aip_python_lib_dir}"
        ARCHIVE DESTINATION "${_aip_python_lib_dir}"
        RUNTIME DESTINATION "${_aip_python_lib_dir}")
    endif()
  endforeach()
endif()

function(aip_add_python_module module_suffix)
  set(module_name "${PROJECT_NAME}_${module_suffix}")

  add_library(${module_name} MODULE "src/${module_suffix}.cpp")

  if(_AIP_RPATH)
    set_target_properties(
      ${module_name} PROPERTIES BUILD_RPATH "${_AIP_RPATH}" INSTALL_RPATH
                                                            "${_AIP_RPATH}")
  endif()

  target_include_directories(
    ${module_name}
    PUBLIC ${BoostPython_INCLUDE_DIRS}
           ${accelerated_image_processor_common_INCLUDE_DIRS}
           ${accelerated_image_processor_compression_INCLUDE_DIRS})

  target_link_libraries(
    ${module_name}
    Boost::python
    Python3::Python
    accelerated_image_processor_common::accelerated_image_processor_common
    accelerated_image_processor_compression::accelerated_image_processor_compression
  )
  set_target_properties(
    ${module_name}
    PROPERTIES PREFIX ""
               SUFFIX ".so"
               BUILD_WITH_INSTALL_RPATH TRUE)

  list(APPEND PYTHON_MODULE_LIST ${module_name})
  set(PYTHON_MODULE_LIST
      "${PYTHON_MODULE_LIST}"
      PARENT_SCOPE)
endfunction()

set(PYTHON_MODULE_LIST)

aip_add_python_module(common)
aip_add_python_module(compression)

install(TARGETS ${PYTHON_MODULE_LIST}
        LIBRARY DESTINATION "${PYTHON_INSTALL_DIR}/${PYTHON_PACKAGE_NAME}")

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  set(test_files test/test_common.py test/test_compression.py)
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
