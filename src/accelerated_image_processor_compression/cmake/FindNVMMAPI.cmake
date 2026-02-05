# ##############################################################################
#
# Original Copyright
#
# ##############################################################################
# Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# - Try to find the NVIDIA Tegra Multimedia API
# Once done this will define
#  NVMMAPI_FOUND - System has NVMMAPI
#  NVMMAPI_INCLUDE_DIRS - The NVMMAPI include directories
#  NVMMAPI_LIBRARIES - The libraries needed to use the NVMMAPI
#  NVMMAPI_DEFINITIONS - Compiler switches required for using NVMMAPI
# ##############################################################################
#
# Modifications Copyright
#
# ##############################################################################
# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
# ##############################################################################

find_package(PkgConfig)

find_path(NVMMAPI_INCLUDE_DIR NvVideoEncoder.h
          HINTS ${SYS_ROOT}/usr/src/jetson_multimedia_api/include)

find_library(
  NVMMAPI_NVV4L2_LIBRARY
  NAMES nvv4l2
  HINTS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/tegra)
find_library(
  NVMMAPI_NVMEDIA_LIBRARY
  NAMES nvmedia
  HINTS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/tegra)
find_library(
  NVMMAPI_V4L2_NVVIDEOCODEC_LIBRARY
  NAMES v4l2_nvvideocodec
  HINTS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/tegra)
find_library(
  NVMMAPI_NVBUFSURFACE_LIBRARY
  NAMES nvbufsurface
  HINTS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/tegra)
find_library(
  NVMMAPI_NVBUFSURF_TRANSFORM_LIBRARY
  NAMES nvbufsurftransform
  HINTS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/tegra)

set(NVMMAPI_LIBRARIES
    ${NVMMAPI_NVV4L2_LIBRARY} ${NVMMAPI_NVMEDIA_LIBRARY}
    ${NVMMAPI_V4L2_NVVIDEOCODEC_LIBRARY} ${NVMMAPI_NVBUFSURFACE_LIBRARY}
    ${NVMMAPI_NVBUFSURF_TRANSFORM_LIBRARY})

set(NVMMAPI_INCLUDE_DIRS ${NVMMAPI_INCLUDE_DIR})
set(NVMMAPI_DEFINITIONS -DNVMMAPI_SUPPORTED)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set ARGUS_FOUND to TRUE if all
# listed variables are TRUE
find_package_handle_standard_args(
  NVMMAPI
  DEFAULT_MSG
  NVMMAPI_INCLUDE_DIR
  NVMMAPI_NVV4L2_LIBRARY
  NVMMAPI_NVMEDIA_LIBRARY
  NVMMAPI_V4L2_NVVIDEOCODEC_LIBRARY
  NVMMAPI_NVBUFSURFACE_LIBRARY
  NVMMAPI_NVBUFSURF_TRANSFORM_LIBRARY)

# mark_as_advanced(NVMMAPI_INCLUDE_DIR NVMMAPI_LIBRARY)
mark_as_advanced(
  NVMMAPI_INCLUDE_DIR NVMMAPI_NVV4L2_LIBRARY NVMMAPI_NVMEDIA_LIBRARY
  NVMMAPI_V4L2_NVVIDEOCODEC_LIBRARY NVMMAPI_NVBUFSURFACE_LIBRARY
  NVMMAPI_NVBUFSURF_TRANSFORM_LIBRARY)
