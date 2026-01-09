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

#include "utility.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

namespace accelerated_image_processor::pipeline
{
common::CameraInfo compute_maps(
  const common::CameraInfo & info, float * map_x, float * map_y, double alpha)
{
  cv::Mat intrinsics(3, 3, CV_64F);
  cv::Mat distortion_coefficients(1, info.d.size(), CV_64F);

  for (size_t row = 0; row < 3; row++) {
    for (size_t col = 0; col < 3; col++) {
      intrinsics.at<double>(row, col) = info.k[row * 3 + col];
    }
  }

  for (std::size_t col = 0; col < info.d.size(); col++) {
    distortion_coefficients.at<double>(col) = info.d[col];
  }

  cv::Mat new_intrinsics = cv::getOptimalNewCameraMatrix(
    intrinsics, distortion_coefficients, cv::Size(info.width, info.height), alpha);

  cv::Mat m1(info.height, info.width, CV_32FC1, map_x);
  cv::Mat m2(info.height, info.width, CV_32FC1, map_y);

  cv::initUndistortRectifyMap(
    intrinsics, distortion_coefficients, cv::Mat::eye(3, 3, CV_64F), new_intrinsics,
    cv::Size(info.width, info.height), CV_32FC1, m1, m2);

  // Copy the original camera info and update only D and K
  common::CameraInfo camera_info_rect(info);
  // After undistortion, the result will be as if it is captured with a camera using
  // the camera with new_intrinsics and zero distortion
  camera_info_rect.d.assign(info.d.size(), 0.);
  for (size_t row = 0; row < 3; row++) {
    for (size_t col = 0; col < 3; col++) {
      camera_info_rect.k[row * 3 + col] = new_intrinsics.at<double>(row, col);
      camera_info_rect.p[row * 4 + col] = new_intrinsics.at<double>(row, col);
    }
  }
  return camera_info_rect;
}
}  // namespace accelerated_image_processor::pipeline
