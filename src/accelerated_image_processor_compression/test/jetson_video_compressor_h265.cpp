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
using H265ParamCombination = std::tuple<
  std::string /* h265.profile */, std::string /* h265.level */, std::string /* compression_type */>;
using TestH265Compressor = TestVideoCompressor<H265ParamCombination>;

TEST_F(TestH265Compressor, JetsonVideoCompressorH265Default)
{
  auto compressor = make_jetson_h265_compressor();
  constexpr int desired_i_frame_interval = 10;

  compressor->register_postprocess<
    TestH265Compressor,
    &TestH265Compressor::check<common::ImageFormat::H265, desired_i_frame_interval>>(this);

  for (auto & [name, value] : compressor->parameters()) {
    if (name == "i_frame_interval") {
      value = desired_i_frame_interval;
    }
  }

  EXPECT_EQ(compressor->parameter_value<int>("i_frame_interval"), desired_i_frame_interval);

  for (auto i = 0; i < TestH265Compressor::NUM_FRAMES; i++) {
    compressor->process(get_image());
  }
}

TEST_P(TestH265Compressor, JetsonVideoCompressorH265ProfileLevelTypeCombo)
{
  auto [profile, level, type] = GetParam();
  auto compressor = make_jetson_h265_compressor();
  constexpr int desired_i_frame_interval = 10;
  compressor->register_postprocess<
    TestH265Compressor,
    &TestH265Compressor::check<common::ImageFormat::H265, desired_i_frame_interval>>(this);

  for (auto & [name, value] : compressor->parameters()) {
    if (name == "i_frame_interval") {
      value = desired_i_frame_interval;
    } else if (name == "h265.profile") {
      value = profile;
    } else if (name == "h265.level") {
      value = level;
    } else if (name == "compression_type") {
      value = type;
    }
  }
  EXPECT_EQ(compressor->parameter_value<int>("i_frame_interval"), desired_i_frame_interval);
  EXPECT_EQ(compressor->parameter_value<std::string>("h265.profile"), profile);
  EXPECT_EQ(compressor->parameter_value<std::string>("h265.level"), level);
  EXPECT_EQ(compressor->parameter_value<std::string>("compression_type"), type);

  for (auto i = 0; i < TestH265Compressor::NUM_FRAMES; i++) {
    compressor->process(get_image());
  }
}

INSTANTIATE_TEST_SUITE_P(
  JestsonVideoCompressorH265Combo, TestH265Compressor,
  ::testing::Combine(
    // Available profiles
    ::testing::Values("MAIN", "MAIN10"),
    // Avaiable levels
    ::testing::Values(
      "1_0_MAIN_TIER", "1_0_HIGH_TIER", "2_0_MAIN_TIER", "2_0_HIGH_TIER", "2_1_MAIN_TIER",
      "2_1_HIGH_TIER", "3_0_MAIN_TIER", "3_0_HIGH_TIER", "3_1_MAIN_TIER", "3_1_HIGH_TIER",
      "4_0_MAIN_TIER", "4_0_HIGH_TIER", "4_1_MAIN_TIER", "4_1_HIGH_TIER", "5_0_MAIN_TIER",
      "5_0_HIGH_TIER", "5_1_MAIN_TIER", "5_1_HIGH_TIER", "5_2_MAIN_TIER", "5_2_HIGH_TIER",
      "6_0_MAIN_TIER", "6_0_HIGH_TIER", "6_1_MAIN_TIER", "6_1_HIGH_TIER", "6_2_MAIN_TIER",
      "6_2_HIGH_TIER"),
    // lossy or lossless
    ::testing::Values("lossy", "lossless")));

}  // namespace accelerated_image_processor::compression
#else
TEST(JetsonVideoCompressorH265Skip, JetsonUnavailable)
{
  GTEST_SKIP() << "Jetson not available. Skipping JetsonVideoCompressorH265 tests.";
}
#endif
int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
