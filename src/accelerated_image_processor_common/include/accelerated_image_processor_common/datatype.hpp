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
 * @brief Enumeration of image color encodings.
 */
enum class ImageEncoding : uint8_t { RGB, BGR };

/**
 * @brief Enumeration of image compression formats.
 */
enum class ImageFormat : uint8_t { RAW, JPEG, PNG };

/**
 * @brief Structure representing an image.
 */
struct Image
{
  std::string frame_id;       //!< Camera frame ID
  int64_t timestamp;          //!< Timestamp at the image is captured
  uint32_t height;            //!< Image height, that is, number of rows
  uint32_t width;             //!< Image width, that is, number of columns
  uint32_t step;              //!< Full row length in bytes
  ImageEncoding encoding;     //!< Image color encoding
  ImageFormat format;         //!< Image compression format
  std::vector<uint8_t> data;  //!< Actual matrix data

  /**
   * @brief Check the specified image is valid.
   */
  bool is_valid() const { return data.size() != 0; }
};

/**
 * @brief Enumeration of distortion models.
 */
enum class DistortionModel : uint8_t { PLUMB_BOB, RATIONAL_POLYNOMIAL, EQUIDISTANT };

/**
 * @brief Region of interest.
 */
struct Roi
{
  uint32_t x_offset;  //!< X offset of the ROI
  uint32_t y_offset;  //!< Y offset of the ROI
  uint32_t width;     //!< Width of the ROI
  uint32_t height;    //!< Height of the ROI
  bool do_rectify;    //!< Whether to rectify the ROI
};

/**
 * @brief Camera information.
 * @todo Implement CameraInfo to common package.
 */
struct CameraInfo
{
  std::string frame_id;              //!< Camera frame ID
  int64_t timestamp;                 //!< Timestamp at the image is captured
  uint32_t height;                   //!< Image height, that is, number of rows
  uint32_t width;                    //!< Image width, that is, number of columns
  DistortionModel distortion_model;  //!< Distortion model
  std::vector<double> d;             //!< Distortion coefficients.
  std::array<double, 9> k;           //!< Intrinsic camera matrix.
  std::array<double, 9> r;           //!< Rectification matrix.
  std::array<double, 12> p;          //!< Extrinsic camera matrix.
  uint32_t binning_x;                //!< Binning factor in x direction
  uint32_t binning_y;                //!< Binning factor in y direction
  Roi roi;                           //!< Region of interest
};
}  // namespace accelerated_image_processor::common
