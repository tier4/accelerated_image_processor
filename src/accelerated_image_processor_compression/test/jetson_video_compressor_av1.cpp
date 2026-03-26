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
using AV1ParamCombination = std::tuple<
  bool /* av1.enable_tile */, int /* av1.log2_num_tile_row */, int /* av1.log2_num_tile_col */,
  bool /* av1.enable_ssim_rdo */, bool /* av1.enable_cdf_update */,
  std::string /* compression_type  */>;
using TestAV1Compressor = TestVideoCompressor<AV1ParamCombination>;

TEST_F(TestAV1Compressor, JetsonVideoCompressorAV1Default)
{
  auto compressor = make_jetson_av1_compressor();
  constexpr int desired_i_frame_interval = 10;

  compressor->register_postprocess<
    TestAV1Compressor,
    &TestAV1Compressor::check<common::ImageFormat::AV1, desired_i_frame_interval>>(this);

  for (auto & [name, value] : compressor->parameters()) {
    if (name == "i_frame_interval") {
      value = desired_i_frame_interval;
    }
  }

  EXPECT_EQ(compressor->parameter_value<int>("i_frame_interval"), desired_i_frame_interval);

  for (auto i = 0; i < TestAV1Compressor::NUM_FRAMES; i++) {
    compressor->process(get_image());
  }
}

TEST_P(TestAV1Compressor, JetsonVideoCompressorAV1ProfileLevelTypeCombo)
{
  auto
    [enable_tile, log2_num_tile_row, log2_num_tile_col, enable_ssim_rdo, enable_cdf_update, type] =
      GetParam();
  auto compressor = make_jetson_av1_compressor();
  constexpr int desired_i_frame_interval = 10;
  compressor->register_postprocess<
    TestAV1Compressor,
    &TestAV1Compressor::check<common::ImageFormat::AV1, desired_i_frame_interval>>(this);

  for (auto & [name, value] : compressor->parameters()) {
    if (name == "i_frame_interval") {
      value = desired_i_frame_interval;
    } else if (name == "av1.enable_tile") {
      value = enable_tile;
    } else if (name == "av1.log2_num_tile_row") {
      value = log2_num_tile_row;
    } else if (name == "av1.log2_num_tile_col") {
      value = log2_num_tile_col;
    } else if (name == "av1.enable_ssim_rdo") {
      value = enable_ssim_rdo;
    } else if (name == "av1.enable_cdf_update") {
      value = enable_cdf_update;
    } else if (name == "compression_type") {
      value = type;
    }
  }
  EXPECT_EQ(compressor->parameter_value<int>("i_frame_interval"), desired_i_frame_interval);
  EXPECT_EQ(compressor->parameter_value<bool>("av1.enable_tile"), enable_tile);
  EXPECT_EQ(compressor->parameter_value<int>("av1.log2_num_tile_row"), log2_num_tile_row);
  EXPECT_EQ(compressor->parameter_value<int>("av1.log2_num_tile_col"), log2_num_tile_col);
  EXPECT_EQ(compressor->parameter_value<bool>("av1.enable_ssim_rdo"), enable_ssim_rdo);
  EXPECT_EQ(compressor->parameter_value<bool>("av1.enable_cdf_update"), enable_cdf_update);

  for (auto i = 0; i < TestAV1Compressor::NUM_FRAMES; i++) {
    compressor->process(get_image());
  }
}

INSTANTIATE_TEST_SUITE_P(
  JetsonVideoCompressorAV1ComboWithTiling, TestAV1Compressor,
  ::testing::Combine(
    // Enable tiling
    ::testing::Values(true),
    // log2 num tile row
    ::testing::Values(
      1, 2),  // NOTE: Judging from the actual behavior, value >= 3 does not seem to be supported
    // log2 num tile col
    ::testing::Values(
      1, 2),  // NOTE: Judging from the actual behavior, value >= 3 does not seem to be supported
    // enable_ssim_rdo
    ::testing::Bool(),
    // enable_cdf_update
    ::testing::Bool(),
    // lossy or lossless
    ::testing::Values("lossy", "lossless")));

INSTANTIATE_TEST_SUITE_P(
  JetsonVideoCompressorAV1ComboWithoutTiling, TestAV1Compressor,
  ::testing::Combine(
    // Disable tiling
    ::testing::Values(false),
    // log2 num tile row (ignored)
    ::testing::Values(0),
    // log2 num tile col (ignored)
    ::testing::Values(0),
    // enable_ssim_rdo
    ::testing::Bool(),
    // enable_cdf_update
    ::testing::Bool(),
    // lossy or lossless
    ::testing::Values("lossy", "lossless")));

}  // namespace accelerated_image_processor::compression
#else
TEST(JetsonVideoCompressorAV1Skip, JetsonUnavailable)
{
  GTEST_SKIP() << "Jetson not available. Skipping JetsonVideoCompressorAV1 tests.";
}
#endif
int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
