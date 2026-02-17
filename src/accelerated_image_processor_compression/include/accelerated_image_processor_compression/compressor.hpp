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

#include <accelerated_image_processor_common/parameter.hpp>
#include <accelerated_image_processor_common/processor.hpp>

namespace accelerated_image_processor::compression
{

/**
 * @enum CompressorBackend
 * @brief Compression backend type.
 *
 * @value JETSON  Jetson hardware accelerated backend.
 * @value NVJPEG  NVIDIA NVJPEG backend.
 * @value CPU     CPU based backend.
 */
enum class CompressorBackend : uint8_t { JETSON, NVJPEG, CPU };

/**
 * @brief Compression type enum
 */
enum class CompressionType : uint8_t { JPEG, H264, H265, AV1 };

/**
 * @brief Base abstract class for image compression processors.
 *
 * The Compressor class provides a common interface for different compression backends
 * such as Jetson hardware accelerated, NVIDIA NVJPEG, and CPU-based implementations.
 * It inherits from common::BaseProcessor and stores the selected backend type.
 *
 * @param backend The compression backend to use.
 * @param dedicated_parameters Optional map of parameters specific to the compressor.
 *
 * @note The class is intended to be subclassed by concrete compressor implementations.
 */
class Compressor : public common::BaseProcessor
{
public:
  explicit Compressor(CompressorBackend backend, common::ParameterMap dedicated_parameters = {})
  : BaseProcessor(dedicated_parameters), backend_(backend)
  {
  }

  virtual ~Compressor() {}

  /**
   * @brief Get the compression backend type.
   *
   * @return The backend type used by this compressor.
   */
  CompressorBackend backend() const { return backend_; }

private:
  const CompressorBackend backend_;  //!< Compression backend type
};

}  // namespace accelerated_image_processor::compression
