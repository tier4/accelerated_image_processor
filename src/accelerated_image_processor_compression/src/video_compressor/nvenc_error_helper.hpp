// Copyright 2026 TIER IV, Inc.
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
#include "enc_result.hpp"

#include <iostream>
#include <string>
#include <string_view>

#ifdef NVENC_AVAILABLE
#include <cuda.h>  // CUDA driver API, which NVENC requires to own the encode device
#include <cuda_runtime.h>
#include <ffnvcodec/nvEncodeAPI.h>
#include <nppdefs.h>

namespace accelerated_image_processor::compression
{
/**
 * @brief Convert NVENCSTATUS into its readable name
 */
constexpr std::string_view nvencstatus_to_string(const NVENCSTATUS status)
{
  switch (status) {
    case NV_ENC_SUCCESS:
      return "NV_ENC_SUCCESS";
    case NV_ENC_ERR_NO_ENCODE_DEVICE:
      return "NV_ENC_ERR_NO_ENCODE_DEVICE";
    case NV_ENC_ERR_UNSUPPORTED_DEVICE:
      return "NV_ENC_ERR_UNSUPPORTED_DEVICE";
    case NV_ENC_ERR_INVALID_ENCODERDEVICE:
      return "NV_ENC_ERR_INVALID_ENCODERDEVICE";
    case NV_ENC_ERR_INVALID_DEVICE:
      return "NV_ENC_ERR_INVALID_DEVICE";
    case NV_ENC_ERR_DEVICE_NOT_EXIST:
      return "NV_ENC_ERR_DEVICE_NOT_EXIST";
    case NV_ENC_ERR_INVALID_PTR:
      return "NV_ENC_ERR_INVALID_PTR";
    case NV_ENC_ERR_INVALID_EVENT:
      return "NV_ENC_ERR_INVALID_EVENT";
    case NV_ENC_ERR_INVALID_PARAM:
      return "NV_ENC_ERR_INVALID_PARAM";
    case NV_ENC_ERR_INVALID_CALL:
      return "NV_ENC_ERR_INVALID_CALL";
    case NV_ENC_ERR_OUT_OF_MEMORY:
      return "NV_ENC_ERR_OUT_OF_MEMORY";
    case NV_ENC_ERR_ENCODER_NOT_INITIALIZED:
      return "NV_ENC_ERR_ENCODER_NOT_INITIALIZED";
    case NV_ENC_ERR_UNSUPPORTED_PARAM:
      return "NV_ENC_ERR_UNSUPPORTED_PARAM";
    case NV_ENC_ERR_LOCK_BUSY:
      return "NV_ENC_ERR_LOCK_BUSY";
    case NV_ENC_ERR_NOT_ENOUGH_BUFFER:
      return "NV_ENC_ERR_NOT_ENOUGH_BUFFER";
    case NV_ENC_ERR_INVALID_VERSION:
      return "NV_ENC_ERR_INVALID_VERSION";
    case NV_ENC_ERR_MAP_FAILED:
      return "NV_ENC_ERR_MAP_FAILED";
    case NV_ENC_ERR_NEED_MORE_INPUT:
      return "NV_ENC_ERR_NEED_MORE_INPUT";
    case NV_ENC_ERR_ENCODER_BUSY:
      return "NV_ENC_ERR_ENCODER_BUSY";
    case NV_ENC_ERR_EVENT_NOT_REGISTERD:
      return "NV_ENC_ERR_EVENT_NOT_REGISTERD";
    case NV_ENC_ERR_GENERIC:
      return "NV_ENC_ERR_GENERIC";
    case NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY:
      return "NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY";
    case NV_ENC_ERR_UNIMPLEMENTED:
      return "NV_ENC_ERR_UNIMPLEMENTED";
    case NV_ENC_ERR_RESOURCE_REGISTER_FAILED:
      return "NV_ENC_ERR_RESOURCE_REGISTER_FAILED";
    case NV_ENC_ERR_RESOURCE_NOT_REGISTERED:
      return "NV_ENC_ERR_RESOURCE_NOT_REGISTERED";
    case NV_ENC_ERR_RESOURCE_NOT_MAPPED:
      return "NV_ENC_ERR_RESOURCE_NOT_MAPPED";
    case NV_ENC_ERR_NEED_MORE_OUTPUT:
      return "NV_ENC_ERR_NEED_MORE_OUTPUT";
    default:
      return "Undefined NVENCSTATUS";
  }
}

/**
 * @brief Compose an error message and dump it to the standard error output
 */
inline EncStatus record_api_error(
  const std::string & what, const std::string & reason, const char * file, const int line)
{
  const std::string msg =
    what + " (" + reason + ") at " + std::string(file) + ":" + std::to_string(line);
  std::cerr << msg << std::endl;
  return EncStatus{false, msg};
}

/**
 * @brief Wraps a NvEncodeAPI call, which returns NV_ENC_SUCCESS on success
 */
inline EncStatus check_nvenc_api_call(
  const NVENCSTATUS status, const char * what, const char * file, const int line)
{
  if (status == NV_ENC_SUCCESS) return EncStatus{true};
  return record_api_error(what, std::string(nvencstatus_to_string(status)), file, line);
}

/**
 * @brief Wraps a CUDA runtime API call, which returns cudaSuccess on success
 */
inline EncStatus check_cuda_api_call(
  const cudaError_t status, const char * what, const char * file, const int line)
{
  if (status == cudaSuccess) return EncStatus{true};
  return record_api_error(what, std::string(cudaGetErrorName(status)), file, line);
}

/**
 * @brief Wraps a CUDA driver API call, which returns CUDA_SUCCESS on success
 */
inline EncStatus check_cu_api_call(
  const CUresult status, const char * what, const char * file, const int line)
{
  if (status == CUDA_SUCCESS) return EncStatus{true};
  const char * name = nullptr;
  cuGetErrorName(status, &name);
  return record_api_error(what, name ? std::string(name) : std::to_string(status), file, line);
}

/**
 * @brief Wraps an NPP call, which returns NPP_SUCCESS on success
 */
inline EncStatus check_npp_api_call(
  const NppStatus status, const char * what, const char * file, const int line)
{
  if (status == NPP_SUCCESS) return EncStatus{true};
  return record_api_error(what, "NppStatus " + std::to_string(status), file, line);
}
}  // namespace accelerated_image_processor::compression

/**
 * @brief Helper macros so that the caller need only write: NVENC_CHECK(f, "Some operation")
 *
 * Each macro returns EncResult carrying the error message from the enclosing function when the
 * wrapped call fails, hence they are usable only in functions that return EncResult.
 */
#define NVENC_CHECK_IMPL(checker, fn, msg)                                                      \
  {                                                                                             \
    auto _res = accelerated_image_processor::compression::checker(fn, msg, __FILE__, __LINE__); \
    if (!_res.ok) {                                                                             \
      return accelerated_image_processor::compression::EncResult{_res};                         \
    }                                                                                           \
  }

#define NVENC_CHECK(fn, msg) NVENC_CHECK_IMPL(check_nvenc_api_call, fn, msg)
#define NVENC_CHECK_CUDA(fn, msg) NVENC_CHECK_IMPL(check_cuda_api_call, fn, msg)
#define NVENC_CHECK_CU(fn, msg) NVENC_CHECK_IMPL(check_cu_api_call, fn, msg)
#define NVENC_CHECK_NPP(fn, msg) NVENC_CHECK_IMPL(check_npp_api_call, fn, msg)
#endif  // NVENC_AVAILABLE
