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

#include "accelerated_image_processor_compression/video_compressor.hpp"
#include "test_utility.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <tuple>

#ifdef JETSON_AVAILABLE
namespace accelerated_image_processor::compression
{
using H264ParamCombination = std::tuple<
  std::string /* h264.profile */, std::string /* h264.level */, std::string /* compression_type */>;
using TestH264Compressor = TestVideoCompressor<H264ParamCombination>;

TEST_F(TestH264Compressor, JetsonVideoCompressorH264Default)
{
  auto compressor = make_jetson_h264_compressor();
  constexpr int desired_i_frame_interval = 10;

  compressor->register_postprocess<
    TestH264Compressor,
    &TestH264Compressor::check<common::ImageFormat::H264, desired_i_frame_interval>>(this);

  for (auto & [name, value] : compressor->parameters()) {
    if (name == "i_frame_interval") {
      value = desired_i_frame_interval;
    }
  }

  EXPECT_EQ(compressor->parameter_value<int>("i_frame_interval"), desired_i_frame_interval);

  for (auto i = 0; i < TestH264Compressor::NUM_FRAMES; i++) {
    compressor->process(get_image());
  }
}

TEST_P(TestH264Compressor, JetsonVideoCompressorH264ProfileLevelTypeCombo)
{
  auto [profile, level, type] = GetParam();
  auto compressor = make_jetson_h264_compressor();
  constexpr int desired_i_frame_interval = 10;
  compressor->register_postprocess<
    TestH264Compressor,
    &TestH264Compressor::check<common::ImageFormat::H264, desired_i_frame_interval>>(this);

  for (auto & [name, value] : compressor->parameters()) {
    if (name == "i_frame_interval") {
      value = desired_i_frame_interval;
    } else if (name == "h264.profile") {
      value = profile;
    } else if (name == "h264.level") {
      value = level;
    } else if (name == "compression_type") {
      value = type;
    }
  }
  EXPECT_EQ(compressor->parameter_value<int>("i_frame_interval"), desired_i_frame_interval);
  EXPECT_EQ(compressor->parameter_value<std::string>("h264.profile"), profile);
  EXPECT_EQ(compressor->parameter_value<std::string>("h264.level"), level);
  EXPECT_EQ(compressor->parameter_value<std::string>("compression_type"), type);

  for (auto i = 0; i < TestH264Compressor::NUM_FRAMES; i++) {
    compressor->process(get_image());
  }
}

INSTANTIATE_TEST_SUITE_P(
  JestsonVideoCompressorH264Combo, TestH264Compressor,
  ::testing::Combine(
    // Available profiles
    ::testing::Values("BASELINE", "MAIN", "HIGH"),
    // Avaiable levels
    ::testing::Values(
      "1_0", "1B", "1_1", "1_2", "1_3", "2_0", "2_1", "2_2", "3_0", "3_1", "3_2", "4_0", "4_1",
      "4_2", "5_0", "5_1"),
    // lossy or lossless
    ::testing::Values("lossy", "lossless")));

}  // namespace accelerated_image_processor::compression
#else
TEST(JetsonVideoCompressorH264Skip, JetsonUnavailable)
{
  GTEST_SKIP() << "Jetson not available. Skipping JetsonVideoCompressorH264 tests.";
}
#endif
int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
