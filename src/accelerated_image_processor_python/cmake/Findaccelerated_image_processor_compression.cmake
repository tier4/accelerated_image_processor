get_filename_component(_aip_src_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(_aip_compression_src
    "${_aip_src_root}/accelerated_image_processor_compression")

# Attempt to reuse an existing target before building from source
if(TARGET
   accelerated_image_processor_compression::accelerated_image_processor_compression
   OR TARGET accelerated_image_processor_compression)
  set(accelerated_image_processor_compression_FOUND TRUE)

  if(NOT DEFINED accelerated_image_processor_compression_INCLUDE_DIRS
     AND TARGET accelerated_image_processor_compression)
    get_target_property(
      _aip_compression_includes accelerated_image_processor_compression
      INTERFACE_INCLUDE_DIRECTORIES)
    set(_aip_compression_include_dirs)
    if(_aip_compression_includes)
      list(APPEND _aip_compression_include_dirs ${_aip_compression_includes})
    endif()
    list(APPEND _aip_compression_include_dirs "${_aip_compression_src}/include")
    list(REMOVE_DUPLICATES _aip_compression_include_dirs)
    set(accelerated_image_processor_compression_INCLUDE_DIRS
        "${_aip_compression_include_dirs}")
  endif()

  return()
endif()

# Locate the sibling package sources (paths computed above to allow reuse before
# dependency discovery)

if(NOT EXISTS "${_aip_compression_src}/CMakeLists.txt")
  set(accelerated_image_processor_compression_FOUND FALSE)
  set(accelerated_image_processor_compression_NOT_FOUND_MESSAGE
      "Could not locate accelerated_image_processor_compression sources at ${_aip_compression_src}"
  )
  return()
endif()

# Build the package from source as part of the current build tree
add_subdirectory(
  "${_aip_compression_src}"
  "${CMAKE_BINARY_DIR}/accelerated_image_processor_compression"
  EXCLUDE_FROM_ALL)

# Provide a namespaced alias to match the expected imported target
if(TARGET accelerated_image_processor_compression
   AND NOT
       TARGET
       accelerated_image_processor_compression::accelerated_image_processor_compression
)
  add_library(
    accelerated_image_processor_compression::accelerated_image_processor_compression
    ALIAS
    accelerated_image_processor_compression)
endif()

get_target_property(
  _aip_compression_includes accelerated_image_processor_compression
  INTERFACE_INCLUDE_DIRECTORIES)
set(_aip_compression_include_dirs)
if(_aip_compression_includes)
  list(APPEND _aip_compression_include_dirs ${_aip_compression_includes})
endif()
list(APPEND _aip_compression_include_dirs "${_aip_compression_src}/include")
list(REMOVE_DUPLICATES _aip_compression_include_dirs)
set(accelerated_image_processor_compression_INCLUDE_DIRS
    "${_aip_compression_include_dirs}")

set(accelerated_image_processor_compression_FOUND TRUE)
