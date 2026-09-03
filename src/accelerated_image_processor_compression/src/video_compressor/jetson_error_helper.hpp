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

#include <cerrno>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>

namespace accelerated_image_processor::compression
{
/**
 * @brief Wraps a NvEncoder/V4L2 call that returns 0 on success and -1 on failure.
 *
 * The macro passes __FILE__ and __LINE__ automatically.
 */
inline EncStatus check_nvenc_call(int fn, const char * file, int line)
{
  int ret = fn;
  if (ret == 0) return EncStatus{true};
  std::string err = std::strerror(errno);
  std::string msg = std::string(file) + ":" + std::to_string(line) + " (" + err + ")";
  std::cerr << msg << std::endl;
  return EncStatus{false, msg};
}

}  // namespace accelerated_image_processor::compression

/**
 * @brief Helper macro so the caller need only write: CHECK_NVENC(f, "Some operation")
 */
#define CHECK_NVENC(fn, msg)                                                              \
  {                                                                                       \
    auto _res =                                                                           \
      accelerated_image_processor::compression::check_nvenc_call(fn, __FILE__, __LINE__); \
    if (!_res.ok) {                                                                       \
      return accelerated_image_processor::compression::EncResult{_res};                   \
    }                                                                                     \
  }
