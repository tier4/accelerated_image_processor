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

namespace accelerated_image_processor::compression
{
class TestJPEGCompressor : public ::testing::Test
{
public:
  void SetUp() override
  {
    image_.frame_id = frame_id;
    image_.timestamp = timestamp;
    image_.width = width;
    image_.height = height;
    image_.step = step;
    image_.encoding = encoding;
    image_.format = format;
    image_.data.resize(image_.step * image_.height);

    for (uint32_t y = 0; y < image_.height; ++y) {
      for (uint32_t x = 0; x < image_.width; ++x) {
        image_.data[y * image_.step + x * 3 + 0] = static_cast<uint8_t>(x % 256);
        image_.data[y * image_.step + x * 3 + 1] = static_cast<uint8_t>(y % 256);
        image_.data[y * image_.step + x * 3 + 2] = static_cast<uint8_t>((x + y) % 256);
      }
    }
  }

  const std::string frame_id = "camera";
  const int64_t timestamp = 123456789;
  const uint32_t width = 1920;
  const uint32_t height = 1080;
  const uint32_t step = width * 3;
  const common::ImageEncoding encoding = common::ImageEncoding::RGB;
  const common::ImageFormat format = common::ImageFormat::RAW;

  const common::Image & get_image() const { return image_; }

  void check(const common::Image & result)
  {
    EXPECT_EQ(result.frame_id, frame_id);
    EXPECT_EQ(result.timestamp, timestamp);
    EXPECT_EQ(result.height, height);
    EXPECT_EQ(result.width, width);
    EXPECT_EQ(result.step, 0);  // 0 means this value is pointless because it's compressed
    EXPECT_EQ(result.encoding, encoding);
    EXPECT_EQ(result.format, common::ImageFormat::JPEG);
    // expect the compressed data size to be smaller than the original image data size, but not 0
    EXPECT_GT(result.data.size(), 0U);
    EXPECT_LE(result.data.size(), image_.data.size());
  }

private:
  common::Image image_;
};
}  // namespace accelerated_image_processor::compression
