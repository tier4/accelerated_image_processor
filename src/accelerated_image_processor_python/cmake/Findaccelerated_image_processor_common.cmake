get_filename_component(_aip_src_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(_aip_common_src "${_aip_src_root}/accelerated_image_processor_common")

# Attempt to reuse an existing target before building from source
if(TARGET accelerated_image_processor_common::accelerated_image_processor_common
   OR TARGET accelerated_image_processor_common)
  set(accelerated_image_processor_common_FOUND TRUE)

  if(NOT DEFINED accelerated_image_processor_common_INCLUDE_DIRS
     AND TARGET accelerated_image_processor_common)
    get_target_property(_aip_common_includes accelerated_image_processor_common
                        INTERFACE_INCLUDE_DIRECTORIES)
    if(_aip_common_includes)
      set(accelerated_image_processor_common_INCLUDE_DIRS
          "${_aip_common_includes}")
    else()
      set(accelerated_image_processor_common_INCLUDE_DIRS
          "${_aip_common_src}/include")
    endif()
  endif()

  return()
endif()

# Locate the sibling package sources
get_filename_component(_aip_src_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(_aip_common_src "${_aip_src_root}/accelerated_image_processor_common")

if(NOT EXISTS "${_aip_common_src}/CMakeLists.txt")
  set(accelerated_image_processor_common_FOUND FALSE)
  set(accelerated_image_processor_common_NOT_FOUND_MESSAGE
      "Could not locate accelerated_image_processor_common sources at ${_aip_common_src}"
  )
  return()
endif()

# Build the package from source as part of the current build tree
add_subdirectory(
  "${_aip_common_src}" "${CMAKE_BINARY_DIR}/accelerated_image_processor_common"
  EXCLUDE_FROM_ALL)

# Provide a namespaced alias to match the expected imported target
if(TARGET accelerated_image_processor_common
   AND NOT TARGET
       accelerated_image_processor_common::accelerated_image_processor_common)
  add_library(
    accelerated_image_processor_common::accelerated_image_processor_common
    ALIAS accelerated_image_processor_common)
endif()

get_target_property(_aip_common_includes accelerated_image_processor_common
                    INTERFACE_INCLUDE_DIRECTORIES)
if(_aip_common_includes)
  set(accelerated_image_processor_common_INCLUDE_DIRS "${_aip_common_includes}")
else()
  set(accelerated_image_processor_common_INCLUDE_DIRS
      "${_aip_common_src}/include")
endif()

set(accelerated_image_processor_common_FOUND TRUE)
