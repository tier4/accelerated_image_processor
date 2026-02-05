// Copyright 2025 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdio>
#include <iostream>

namespace accelerated_image_processor::common
{
/**
 * @brief Macro to check for errors and print a message if the condition is true.
 *
 * @param cond The condition to check.
 * @param str The message to print if the condition is true.
 */
#define CHECK_ERROR(cond, str)    \
  if (cond) {                     \
    fprintf(stderr, "%s\n", str); \
  }

/**
 * @brief Macro to check for errors and print a message if the NPP status is not NPP_SUCCESS.
 *
 * @param status The NPP status.
 */
#define CHECK_NPP(status)                                                              \
  if (status != NPP_SUCCESS) {                                                         \
    std::cerr << "NPP error: " << status << " (" << __FILE__ << ":" << __LINE__ << ")" \
              << std::endl;                                                            \
  }

/**
 * @brief Macro to check for errors and print a message if the CUDA status is not cudaSuccess.
 *
 * @param status The CUDA status.
 */
#define CHECK_CUDA(status)                                                                         \
  if (status != cudaSuccess) {                                                                     \
    std::cerr << "CUDA error: " << cudaGetErrorName(status) << " (" << __FILE__ << ":" << __LINE__ \
              << ")" << std::endl;                                                                 \
  }

/**
 * @brief Macro to check for errors and print a message if the NVJPEG status is not
 * NVJPEG_STATUS_SUCCESS.
 *
 * @param status The NVJPEG status.
 */
#define CHECK_NVJPEG(call)                                                                \
  {                                                                                       \
    nvjpegStatus_t _e = (call);                                                           \
    if (_e != NVJPEG_STATUS_SUCCESS) {                                                    \
      std::cerr << "NVJPEG failure: \'#" << _e << "\' at " << __FILE__ << ":" << __LINE__ \
                << std::endl;                                                             \
      exit(1);                                                                            \
    }                                                                                     \
  }

/**
 * @brief Macro to check for errors and print a message if the VPI status is not
 * VPI_SUCCESS
 *
 * @param status The VPI status.
 */
#define CHECK_VPI(call)                                                                \
  {                                                                                    \
    VPIStatus _e = (call);                                                             \
    if (_e != VPI_SUCCESS) {                                                           \
      char msg_buf[VPI_MAX_STATUS_MESSAGE_LENGTH];                                     \
      vpiGetLastStatusMessage(msg_buf, VPI_MAX_STATUS_MESSAGE_LENGTH);                 \
      std::cerr << "VPI failure: \'#" << _e << "\' at " << __FILE__ << ":" << __LINE__ \
                << ", (reason: " << std::string(msg_buf) << ")" << std::endl;          \
      exit(1);                                                                         \
    }                                                                                  \
  }

}  // namespace accelerated_image_processor::common
