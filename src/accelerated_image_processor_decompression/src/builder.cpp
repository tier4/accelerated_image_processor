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

#include "accelerated_image_processor_decompression/builder.hpp"

#include "accelerated_image_processor_decompression/video_decompressor.hpp"

#include <algorithm>
#include <cctype>
#include <memory>
#include <stdexcept>
#include <string>

namespace accelerated_image_processor::decompression
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

DecompressionType to_decompression_type(const std::string & str)
{
  const auto s = normalize_str(str);
  if (s == "VIDEO") {
    return DecompressionType::VIDEO;
  } else {
    throw std::invalid_argument("Invalid decompression type: " + str);
  }
}

std::unique_ptr<Decompressor> create_decompressor(DecompressionType type)
{
  switch (type) {
    case DecompressionType::VIDEO:
      return make_ffmpeg_video_decompressor();
    default:
      throw std::invalid_argument("Invalid decompression type");
  }
}

std::unique_ptr<Decompressor> create_decompressor(const std::string & type)
{
  return create_decompressor(to_decompression_type(type));
}
}  // namespace accelerated_image_processor::decompression
