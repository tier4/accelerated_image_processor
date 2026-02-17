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

#include <accelerated_image_processor_common/datatype.hpp>
#include <accelerated_image_processor_common/parameter.hpp>
#include <accelerated_image_processor_common/processor.hpp>
#include <accelerated_image_processor_compression/compressor.hpp>

#include <memory>

namespace accelerated_image_processor::compression
{
class JetsonJPEGCompressor;
class NvJPEGCompressor;
class CpuJPEGCompressor;

/**
 * @brief Abstract base class for JPEG compressors.
 */
class JPEGCompressor : public Compressor
{
public:
  explicit JPEGCompressor(CompressorBackend backend, common::ParameterMap dedicated_parameters = {})
  : Compressor(backend, dedicated_parameters += {{"quality", 90}})
  {
  }

  ~JPEGCompressor() override = default;

  /**
   * @brief Return the quality of the JPEG compression.
   */
  int quality() const { return this->parameter_value<int>("quality"); }
};
//!< @brief Factory function to create a CPUJPEGCompressor.
std::unique_ptr<JPEGCompressor> make_cpujpeg_compressor();
//!< @brief Factory function to create a NvJPEGCompressor.
std::unique_ptr<JPEGCompressor> make_nvjpeg_compressor();
//!< @brief Factory function to create a JetsonJPEGCompressor.
std::unique_ptr<JPEGCompressor> make_jetsonjpeg_compressor();
}  // namespace accelerated_image_processor::compression
