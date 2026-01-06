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

#include "accelerated_image_processor_compression/builder.hpp"

#include "accelerated_image_processor_compression/jpeg_compressor.hpp"

#include <algorithm>
#include <cctype>
#include <memory>
#include <stdexcept>
#include <string>

namespace accelerated_image_processor::compression
{
namespace
{
/**
 * @brief Normalize a string by removing leading and trailing whitespace and converting to
 * uppercase.
 *
 * @param s The string to normalize.
 * @return std::string The normalized string.
 */
inline std::string normalize_str(std::string s)
{
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());

  // upper
  for (auto & c : s) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  return s;
}
}  // namespace

CompressionType to_compression_type(const std::string & str)
{
  const auto s = normalize_str(str);
  if (s == "JPEG") {
    return CompressionType::JPEG;
  } else if (s == "VIDEO") {
    return CompressionType::VIDEO;
  } else {
    throw std::invalid_argument("Invalid compression type: " + str);
  }
}

std::unique_ptr<Compressor> create_compressor(CompressionType type)
{
  switch (type) {
    case CompressionType::JPEG:
#ifdef JETSON_AVAILABLE
      return make_jetsonjpeg_compressor();
#elif NVJPEG_AVAILABLE
      return make_nvjpeg_compressor();
#elif TURBOJPEG_AVAILABLE
      return make_cpujpeg_compressor();
#else
      throw std::runtime_error("No JPEG compressor available");
#endif
    case CompressionType::VIDEO:
      throw std::runtime_error("VIDEO compression is not supported yet");
    default:
      throw std::invalid_argument("Invalid compression type");
  }
}

std::unique_ptr<Compressor> create_compressor(const std::string & type)
{
  return create_compressor(to_compression_type(type));
}
}  // namespace accelerated_image_processor::compression
