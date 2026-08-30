# * Try to find nv-codec-headers (ffnvcodec), which provides the NVIDIA Video
#   Codec SDK headers such as nvEncodeAPI.h.
#
# The headers are resolved in the following order:
#
# Step 1: FFNVCODEC_ROOT_DIR, which defaults to a nv-codec-headers checkout
# placed next to this repository. That is the layout `vcs import` produces for a
# workspace build:
#
# <workspace>/src/accelerated_image_processor <- this repository
# <workspace>/src/nv-codec-headers            <- the headers
#
# Step 2: A system installation, found via pkg-config (ffnvcodec.pc, which `make
# install` of nv-codec-headers installs) or the standard include directories.
#
# Step 3: A download from GitHub via FetchContent, unless
# FFNVCODEC_ALLOW_DOWNLOAD is turned off.
#
# Note that only headers are needed at build time: the implementation lives in
# libnvidia-encode.so, which the NVIDIA display driver installs and which the
# compressor loads at runtime via dlopen().
#
# The following are set after configuration is done: FFNVCODEC_FOUND
# FFNVCODEC_INCLUDE_DIR FFNVCODEC_INCLUDE_DIRS FFNVCODEC_VERSION

# This module sits in <this package>/cmake, hence three levels up is the root of
# this repository and four levels up is the directory that holds the repository
# together with its siblings
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
# NvEncodeAPI is not forward compatible: the display driver has to be equal to
# or newer than the SDK the headers come from (NvEncodeAPI 13.0 requires the
# driver 570.0 or later). Therefore a conservative tag is pinned here, which can
# be overridden via -DFFNVCODEC_DOWNLOAD_TAG when a newer SDK is needed.
set(FFNVCODEC_DOWNLOAD_TAG
    "n13.0.19.1"
    CACHE STRING "git tag of nv-codec-headers to be downloaded")

# Step 1: the checkout next to this repository, or wherever FFNVCODEC_ROOT_DIR
# points at
if(FFNVCODEC_ROOT_DIR)
  find_path(
    FFNVCODEC_INCLUDE_DIR ffnvcodec/nvEncodeAPI.h
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

# Step 2: a system installation
if(NOT FFNVCODEC_INCLUDE_DIR)
  find_package(PkgConfig QUIET)
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(PC_FFNVCODEC QUIET ffnvcodec)
  endif()

  find_path(
    FFNVCODEC_INCLUDE_DIR ffnvcodec/nvEncodeAPI.h
    HINTS ${PC_FFNVCODEC_INCLUDE_DIRS}
    PATH_SUFFIXES include)
endif()

# Step 3: download
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
  # nv-codec-headers ships no CMakeLists.txt, hence this only populates the
  # source tree instead of adding it as a subproject
  FetchContent_MakeAvailable(nv_codec_headers)
  FetchContent_GetProperties(nv_codec_headers
                             SOURCE_DIR nv_codec_headers_source_dir)
  if(EXISTS "${nv_codec_headers_source_dir}/include/ffnvcodec/nvEncodeAPI.h")
    set(FFNVCODEC_INCLUDE_DIR
        "${nv_codec_headers_source_dir}/include"
        CACHE PATH "Path to the ffnvcodec headers" FORCE)
  endif()
endif()

set(FFNVCODEC_VERSION "")
if(EXISTS "${FFNVCODEC_INCLUDE_DIR}/ffnvcodec/nvEncodeAPI.h")
  file(STRINGS "${FFNVCODEC_INCLUDE_DIR}/ffnvcodec/nvEncodeAPI.h"
       FFNVCODEC_VERSION_MAJOR_LINE
       REGEX "^[ \t]*#define[ \t]+NVENCAPI_MAJOR_VERSION[ \t]+[0-9]+")
  file(STRINGS "${FFNVCODEC_INCLUDE_DIR}/ffnvcodec/nvEncodeAPI.h"
       FFNVCODEC_VERSION_MINOR_LINE
       REGEX "^[ \t]*#define[ \t]+NVENCAPI_MINOR_VERSION[ \t]+[0-9]+")
  string(REGEX
         REPLACE ".*NVENCAPI_MAJOR_VERSION[ \t]+([0-9]+).*" "\\1"
                 FFNVCODEC_VERSION_MAJOR "${FFNVCODEC_VERSION_MAJOR_LINE}")
  string(REGEX
         REPLACE ".*NVENCAPI_MINOR_VERSION[ \t]+([0-9]+).*" "\\1"
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
