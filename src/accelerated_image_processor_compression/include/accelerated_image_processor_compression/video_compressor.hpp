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

#include <accelerated_image_processor_common/datatype.hpp>
#include <accelerated_image_processor_common/parameter.hpp>
#include <accelerated_image_processor_common/processor.hpp>
#include <accelerated_image_processor_compression/compressor.hpp>

#include <algorithm>
#include <cctype>
#include <locale>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>

namespace accelerated_image_processor::compression
{
class JetsonVideoCompressor;
class JetsonH264Compressor;
class JetsonH265Compressor;
class JetsonAV1Compressor;

/**
 * @brief Enumeration of video encoder mode and string map
 */
enum class VideoCompressionType : uint8_t { LOSSY, LOSSLESS };
const std::unordered_map<std::string, VideoCompressionType> video_compression_type_map = {
  {"LOSSY", VideoCompressionType::LOSSY}, {"LOSSLESS", VideoCompressionType::LOSSLESS}};

/**
 * @brief Utility function to convert std::string to enum class
 */
template <typename EnumType>
EnumType string_to_enum(
  const std::string & str, const std::unordered_map<std::string, EnumType> & mapping)
{
  std::string uppercase_str = str;
  std::transform(str.begin(), str.end(), uppercase_str.begin(), [](const unsigned char & c) {
    return std::toupper(c);
  });

  auto it = mapping.find(uppercase_str);
  if (it != mapping.end()) {
    return it->second;
  } else {
    throw std::runtime_error("Invalid parameter was set: " + str);
  }
}

/**
 * @brief Abstract base class for Jetson Video compressors.
 */
class VideoCompressor : public Compressor
{
public:
  explicit VideoCompressor(
    CompressorBackend backend, common::ParameterMap dedicated_parameters = {})
  : Compressor(
      backend, dedicated_parameters += {
                 {"compression_type", static_cast<std::string>("lossy")},
                 {"idr_frame_interval", static_cast<int>(10)},
                 {"i_frame_interval", static_cast<int>(10)},
                 {"frame_rate_numerator", static_cast<int>(10)},   // frame
                 {"frame_rate_denominator", static_cast<int>(1)},  // Second
               })
  {
  }

  ~VideoCompressor() override = default;

  /**
   * @brief Validates the compatibility of the current compression parameters.
   *
   * This method checks whether the configured compression settings are compatible
   * with the underlying hardware and software capabilities. It returns a tuple
   * containing a boolean flag indicating success and an optional message
   * describing any incompatibilities.
   *
   * @return std::tuple<bool, std::string> A tuple where the first element is
   * true if the parameters are compatible, false otherwise; the second element
   * contains an explanatory message when the parameters are not compatible.
   */
  virtual std::tuple<bool, std::string> validate_compression_type_compatibility()
  {
    bool is_compatible = true;
    std::string message = "";
    return {is_compatible, message};
  }
};
//!< @brief Factory function to create a JetsonH264Compressor.
std::unique_ptr<VideoCompressor> make_jetson_h264_compressor();
//!< @brief Factory function to create a JetsonH265Compressor.
std::unique_ptr<VideoCompressor> make_jetson_h265_compressor();
//!< @brief Factory function to create a JetsonAV1Compressor.
std::unique_ptr<VideoCompressor> make_jetson_av1_compressor();
}  // namespace accelerated_image_processor::compression
