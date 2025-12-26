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

#include <gtest/gtest.h>

#include <string>

namespace accelerated_image_processor::pipeline
{
class TestRectifier : public ::testing::Test
{
public:
  void SetUp() override
  {
    // initialize image
    image_.frame_id = frame_id;
    image_.timestamp = timestamp;
    image_.width = width;
    image_.height = height;
    image_.step = step;
    image_.format = format;
    image_.data.resize(image_.step * image_.height);

    for (uint32_t y = 0; y < image_.height; ++y) {
      for (uint32_t x = 0; x < image_.width; ++x) {
        image_.data[y * image_.step + x * 3 + 0] = static_cast<uint8_t>(x % 256);
        image_.data[y * image_.step + x * 3 + 1] = static_cast<uint8_t>(y % 256);
        image_.data[y * image_.step + x * 3 + 2] = static_cast<uint8_t>((x + y) % 256);
      }
    }

    // initialize camera info
    camera_info_.frame_id = frame_id;
    camera_info_.timestamp = timestamp;
    camera_info_.width = width;
    camera_info_.height = height;
    camera_info_.d = {0.0, 0.0, 0.0, 0.0, 0.0};  // (k1, k2, p1, p2, k3)
    camera_info_.k = {1.0, 0.0, width / 2.0,     // (fx, 0, cx)
                      0.0, 1.0, height / 2.0,    // (0, fy, cy)
                      0.0, 0.0, 1.0};            // (0, 0, 1.0)
    camera_info_.p = {1.0, 0.0, 0.0, 0.0,        // (r11, r12, r13, tx)
                      0.0, 1.0, 0.0, 0.0,        // (r21, r22, r23, ty)
                      0.0, 0.0, 1.0, 0.0};       // (r31, r32, r33, tz)
  }

  const std::string frame_id = "camera";
  const uint32_t timestamp = 123456789;
  const uint32_t width = 1920;
  const uint32_t height = 1080;
  const uint32_t step = width * 3;
  const common::ImageFormat format = common::ImageFormat::RGB;

  const common::Image & get_image() const { return image_; }
  const common::CameraInfo & get_camera_info() const { return camera_info_; }

  void check(const common::Image & result)
  {
    // check image data
    EXPECT_EQ(result.frame_id, frame_id);
    EXPECT_EQ(result.timestamp, timestamp);
    EXPECT_EQ(result.height, height);
    EXPECT_EQ(result.width, width);
    EXPECT_EQ(result.step, step);
    EXPECT_EQ(result.format, format);
    EXPECT_EQ(result.data.size(), image_.data.size());
  }

private:
  common::Image image_;
  common::CameraInfo camera_info_;
};
}  // namespace accelerated_image_processor::pipeline
