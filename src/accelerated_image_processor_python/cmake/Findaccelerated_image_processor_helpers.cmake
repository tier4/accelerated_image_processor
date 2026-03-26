# Helper utilities shared by Findaccelerated_image_processor_*.cmake modules.
#
# Provides aip_find_sibling_package(<suffix>) which encapsulates the common
# logic used to discover sibling accelerated_image_processor packages when
# building the Python bindings with scikit-build or colcon.

if(COMMAND aip_find_sibling_package)
  return()
endif()

function(_aip_select_existing_target package_name out_var)
  set(_candidate "")
  if(TARGET ${package_name})
    set(_candidate "${package_name}")
  elseif(TARGET ${package_name}::${package_name})
    set(_candidate "${package_name}::${package_name}")
  endif()
  set(${out_var}
      "${_candidate}"
      PARENT_SCOPE)
endfunction()

function(_aip_collect_include_dirs target package_src out_var)
  set(_include_dirs)
  if(target)
    get_target_property(_iface_includes ${target} INTERFACE_INCLUDE_DIRECTORIES)
    if(_iface_includes AND NOT _iface_includes STREQUAL
                           "INTERFACE_INCLUDE_DIRECTORIES-NOTFOUND")
      list(APPEND _include_dirs ${_iface_includes})
    endif()
  endif()
  if(EXISTS "${package_src}/include")
    list(APPEND _include_dirs "${package_src}/include")
  endif()
  list(REMOVE_DUPLICATES _include_dirs)
  set(${out_var}
      "${_include_dirs}"
      PARENT_SCOPE)
endfunction()

function(aip_find_sibling_package suffix)
  if(NOT suffix)
    message(
      FATAL_ERROR
        "aip_find_sibling_package requires a suffix argument (e.g. common, compression)"
    )
  endif()

  set(package_name "accelerated_image_processor_${suffix}")
  set(found_var "${package_name}_FOUND")
  set(not_found_var "${package_name}_NOT_FOUND_MESSAGE")
  set(include_var "${package_name}_INCLUDE_DIRS")

  get_filename_component(_aip_src_root "${CMAKE_CURRENT_LIST_DIR}/../.."
                         ABSOLUTE)
  set(package_src "${_aip_src_root}/${package_name}")

  _aip_select_existing_target(${package_name} _aip_existing_target)
  if(_aip_existing_target)
    _aip_collect_include_dirs("${_aip_existing_target}" "${package_src}"
                              _aip_include_dirs)
    set(${include_var}
        "${_aip_include_dirs}"
        PARENT_SCOPE)
    set(${found_var}
        TRUE
        PARENT_SCOPE)
    return()
  endif()

  if(NOT EXISTS "${package_src}/CMakeLists.txt")
    set(${found_var}
        FALSE
        PARENT_SCOPE)
    set(${not_found_var}
        "Could not locate ${package_name} sources at ${package_src}"
        PARENT_SCOPE)
    return()
  endif()

  add_subdirectory("${package_src}" "${CMAKE_BINARY_DIR}/${package_name}"
                   EXCLUDE_FROM_ALL)

  if(TARGET ${package_name} AND NOT TARGET ${package_name}::${package_name})
    add_library(${package_name}::${package_name} ALIAS ${package_name})
  endif()

  _aip_select_existing_target(${package_name} _aip_new_target)
  _aip_collect_include_dirs("${_aip_new_target}" "${package_src}"
                            _aip_include_dirs)

  set(${include_var}
      "${_aip_include_dirs}"
      PARENT_SCOPE)
  set(${found_var}
      TRUE
      PARENT_SCOPE)
endfunction()
