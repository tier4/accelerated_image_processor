# Try to find nv-codec-headers (ffnvcodec), which provides the NVDEC API headers
# and dynamic-loading utility.
#
# The headers are resolved in the following order:
#
# 1. FFNVCODEC_ROOT_DIR, which defaults to a sibling nv-codec-headers checkout.
# 2. A system installation found through pkg-config or standard include
#    directories.
# 3. A download from GitHub, unless FFNVCODEC_ALLOW_DOWNLOAD is disabled.
#
# The following variables are set: FFNVCODEC_FOUND FFNVCODEC_INCLUDE_DIR
# FFNVCODEC_INCLUDE_DIRS FFNVCODEC_VERSION

get_filename_component(FFNVCODEC_REPOSITORY_ROOT
                       "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
get_filename_component(
  FFNVCODEC_DEFAULT_ROOT_DIR "${FFNVCODEC_REPOSITORY_ROOT}/../nv-codec-headers"
  ABSOLUTE)

set(FFNVCODEC_ROOT_DIR
    "${FFNVCODEC_DEFAULT_ROOT_DIR}"
    CACHE PATH "Folder that contains include/ffnvcodec of nv-codec-headers")
option(FFNVCODEC_ALLOW_DOWNLOAD
       "Download nv-codec-headers when it is not found on the system" ON)
# Version 13.0 remains compatible with NVIDIA display drivers from version 570.
set(FFNVCODEC_DOWNLOAD_TAG
    "n13.0.19.1"
    CACHE STRING "git tag of nv-codec-headers to be downloaded")

if(FFNVCODEC_ROOT_DIR)
  find_path(
    FFNVCODEC_INCLUDE_DIR ffnvcodec/dynlink_loader.h
    PATHS ${FFNVCODEC_ROOT_DIR}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH)

  if(NOT FFNVCODEC_INCLUDE_DIR)
    message(
      STATUS
        "nv-codec-headers not found in ${FFNVCODEC_ROOT_DIR}. Falling back to a system installation..."
    )
  endif()
endif()

if(NOT FFNVCODEC_INCLUDE_DIR)
  find_package(PkgConfig QUIET)
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(PC_FFNVCODEC QUIET ffnvcodec)
  endif()

  find_path(
    FFNVCODEC_INCLUDE_DIR ffnvcodec/dynlink_loader.h
    HINTS ${PC_FFNVCODEC_INCLUDE_DIRS}
    PATH_SUFFIXES include)
endif()

if(NOT FFNVCODEC_INCLUDE_DIR AND FFNVCODEC_ALLOW_DOWNLOAD)
  message(
    STATUS
      "nv-codec-headers not found on the system. Downloading ${FFNVCODEC_DOWNLOAD_TAG} via FetchContent..."
  )
  include(FetchContent)
  FetchContent_Declare(
    nv_codec_headers
    GIT_REPOSITORY https://github.com/FFmpeg/nv-codec-headers.git
    GIT_TAG ${FFNVCODEC_DOWNLOAD_TAG})
  FetchContent_MakeAvailable(nv_codec_headers)
  FetchContent_GetProperties(nv_codec_headers
                             SOURCE_DIR nv_codec_headers_source_dir)
  if(EXISTS "${nv_codec_headers_source_dir}/include/ffnvcodec/dynlink_loader.h")
    set(FFNVCODEC_INCLUDE_DIR
        "${nv_codec_headers_source_dir}/include"
        CACHE PATH "Path to the ffnvcodec headers" FORCE)
  endif()
endif()

set(FFNVCODEC_VERSION "")
if(EXISTS "${FFNVCODEC_INCLUDE_DIR}/ffnvcodec/dynlink_cuviddec.h")
  file(STRINGS "${FFNVCODEC_INCLUDE_DIR}/ffnvcodec/dynlink_cuviddec.h"
       FFNVCODEC_VERSION_MAJOR_LINE
       REGEX "^[ \t]*#define[ \t]+NVDECAPI_MAJOR_VERSION[ \t]+[0-9]+")
  file(STRINGS "${FFNVCODEC_INCLUDE_DIR}/ffnvcodec/dynlink_cuviddec.h"
       FFNVCODEC_VERSION_MINOR_LINE
       REGEX "^[ \t]*#define[ \t]+NVDECAPI_MINOR_VERSION[ \t]+[0-9]+")
  string(REGEX
         REPLACE ".*NVDECAPI_MAJOR_VERSION[ \t]+([0-9]+).*" "\\1"
                 FFNVCODEC_VERSION_MAJOR "${FFNVCODEC_VERSION_MAJOR_LINE}")
  string(REGEX
         REPLACE ".*NVDECAPI_MINOR_VERSION[ \t]+([0-9]+).*" "\\1"
                 FFNVCODEC_VERSION_MINOR "${FFNVCODEC_VERSION_MINOR_LINE}")
  set(FFNVCODEC_VERSION "${FFNVCODEC_VERSION_MAJOR}.${FFNVCODEC_VERSION_MINOR}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  FFNVCODEC
  REQUIRED_VARS FFNVCODEC_INCLUDE_DIR
  VERSION_VAR FFNVCODEC_VERSION)

if(FFNVCODEC_FOUND)
  set(FFNVCODEC_INCLUDE_DIRS ${FFNVCODEC_INCLUDE_DIR})
  mark_as_advanced(FFNVCODEC_INCLUDE_DIR)

  if(NOT TARGET FFNVCODEC::ffnvcodec)
    add_library(FFNVCODEC::ffnvcodec INTERFACE IMPORTED)
    set_target_properties(
      FFNVCODEC::ffnvcodec PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                      "${FFNVCODEC_INCLUDE_DIR}")
  endif()
endif()
