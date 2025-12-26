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

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace accelerated_image_processor::common
{
/**
 * @brief Enumeration of image formats.
 */
enum class ImageFormat : uint8_t { RGB, BGR };

/**
 * @brief Structure representing an image.
 */
struct Image
{
  std::string frame_id;       //!< Camera frame ID
  uint64_t timestamp;         //!< Timestamp at the image is captured
  uint32_t height;            //!< Image height, that is, number of rows
  uint32_t width;             //!< Image width, that is, number of columns
  uint32_t step;              //!< Full row length in bytes
  ImageFormat format;         //!< Image format, either RGB or BGR
  std::vector<uint8_t> data;  //!< Actual matrix data, size is (step * height)
};

/**
 * @brief Camera information.
 * @todo Implement CameraInfo to common package.
 */
struct CameraInfo
{
  std::string frame_id;      //!< Camera frame ID
  uint64_t timestamp;        //!< Timestamp at the image is captured
  uint32_t height;           //!< Image height, that is, number of rows
  uint32_t width;            //!< Image width, that is, number of columns
  std::vector<double> d;     //!< Distortion coefficients.
  std::array<double, 9> k;   //!< Intrinsic camera matrix.
  std::array<double, 12> p;  //!< Extrinsic camera matrix.
};
}  // namespace accelerated_image_processor::common
