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

#include <optional>
#include <stdexcept>
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

template <typename ParamCombination>
class TestVideoCompressor : public ::testing::TestWithParam<ParamCombination>
{
public:
  static constexpr int NUM_FRAMES = 20;

  void SetUp() override
  {
    for (uint32_t i = 0; i < images_.size(); i++) {
      auto & image = images_[i];
      image.frame_id = frame_id;
      // image.timestamp = timestamp + (i * 100'000'000ULL);  // + (i * 100ms)
      image.timestamp = timestamp;
      image.width = width;
      image.height = height;
      image.step = step;
      image.encoding = encoding;
      image.format = format;
      image.data.resize(image.step * image.height);

      for (uint32_t y = 0; y < image.height; ++y) {
        for (uint32_t x = 0; x < image.width; ++x) {
          image.data[y * image.step + x * 3 + 0] = static_cast<uint8_t>((x + i) % 256);
          image.data[y * image.step + x * 3 + 1] = static_cast<uint8_t>((y + i) % 256);
          image.data[y * image.step + x * 3 + 2] = static_cast<uint8_t>((x + y + i) % 256);
        }
      }
    }

    index_ = 0;
    image_size_ = images_[0].data.size();  // Save representative image size
    frame_from_first_i_frame_ = std::nullopt;
  }

  const std::string frame_id = "camera";
  const int64_t timestamp = 123456789;
  const uint32_t width = 1920;
  const uint32_t height = 1080;
  const uint32_t step = width * 3;
  const common::ImageEncoding encoding = common::ImageEncoding::RGB;
  const common::ImageFormat format = common::ImageFormat::RAW;

  const common::Image & get_image()
  {
    // Yield image one by one
    if (index_ < images_.size()) {
      return images_[index_++];
    }
    throw std::out_of_range("TestVideoCompressor's generator exhausted.");
  }

  template <common::ImageFormat Fmt, int IFrameInterval>
  void check(const common::Image & result)
  {
    EXPECT_EQ(result.frame_id, frame_id);
    EXPECT_EQ(result.timestamp, timestamp);
    EXPECT_EQ(result.height, height);
    EXPECT_EQ(result.width, width);
    EXPECT_EQ(result.format, Fmt);
    // expect the compressed data size to be smaller than the original image data size, but not 0
    EXPECT_GT(result.data.size(), 0U);
    EXPECT_LE(result.data.size(), image_size_);
    // expect the pts field has values larger than zero
    EXPECT_TRUE(result.pts.has_value());
    EXPECT_GT(result.pts.value(), 0);
    // expect flag should be 0 or 1
    EXPECT_TRUE(result.flags.has_value());
    if (result.flags.value() == 1 && frame_from_first_i_frame_ == std::nullopt) {
      // The first I frame is observed. start counting
      frame_from_first_i_frame_ = 0;
    } else {
      (*frame_from_first_i_frame_)++;
    }

    if (!frame_from_first_i_frame_) {
      // The first I frame has not been observed.
      EXPECT_EQ(result.flags.value(), 0);
    } else if (
      frame_from_first_i_frame_.has_value() &&
      frame_from_first_i_frame_.value() % IFrameInterval != 0) {
      // The first I frame has been observed, and this frame should not be I frames
      EXPECT_EQ(result.flags.value(), 0);
    } else {
      // This should be the I frame
      EXPECT_EQ(result.flags.value(), 1);
    }
  }

private:
  std::array<common::Image, NUM_FRAMES> images_;
  size_t index_;
  size_t image_size_;
  std::optional<size_t> frame_from_first_i_frame_;
};
}  // namespace accelerated_image_processor::compression
