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

#include <cstddef>
#include <cstdint>
#include <vector>

namespace accelerated_image_processor::common
{
/**
 * @brief Enumeration of image formats.
 */
enum ImageFormat { RGB, BGR };

/**
 * @brief Structure representing an image.
 */
struct Image
{
  size_t height;              //!< Image height, that is, number of rows
  size_t width;               //!< Image width, that is, number of columns
  size_t step;                //!< Full row length in bytes
  ImageFormat format;         //!< Image format, either RGB or BGR
  std::vector<uint8_t> data;  //!< Actual matrix data, size is (step * height)

  /**
   * @brief Get the size of the image in bytes.
   * @return The size of the image in bytes.
   */
  size_t size() const { return height * step; }
};
}  // namespace accelerated_image_processor::common
