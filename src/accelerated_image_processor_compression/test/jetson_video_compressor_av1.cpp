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
#include "video_compressor/av1_obu.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>

#ifdef JETSON_AVAILABLE
namespace accelerated_image_processor::compression
{
using AV1ParamCombination = std::tuple<
  bool /* av1.enable_tile */, int /* av1.log2_num_tile_row */, int /* av1.log2_num_tile_col */,
  bool /* av1.enable_ssim_rdo */, bool /* av1.enable_cdf_update */,
  std::string /* compression_type  */>;
using TestAV1Compressor = TestVideoCompressor<AV1ParamCombination>;

/**
 * @brief Fixture that additionally validates the OBU layout of every emitted AV1 packet
 */
class TestAV1RandomAccess : public TestAV1Compressor
{
public:
  std::atomic<int> num_checked{0};

  template <common::ImageFormat Fmt, int IFrameInterval>
  void check_with_obu_layout(const common::Image & result)
  {
    this->check<Fmt, IFrameInterval>(result);

    const uint8_t * data = result.data.data();
    const size_t size = result.data.size();

    size_t num_temporal_delimiter = 0;
    size_t num_sequence_header = 0;
    size_t num_frame = 0;  // OBU_FRAME_HEADER + OBU_FRAME (i.e. one shown frame each)
    std::optional<av1_obu::ObuType> first_obu_type;
    size_t first_obu_end = 0;
    std::optional<size_t> first_sequence_header_pos;
    std::optional<size_t> first_frame_pos;

    size_t pos = 0;
    while (const auto obu = av1_obu::next_obu(data, size, pos)) {
      if (obu->offset == 0) {
        first_obu_type = obu->type;
        first_obu_end = obu->size;
      }
      switch (obu->type) {
        case av1_obu::ObuType::TEMPORAL_DELIMITER:
          num_temporal_delimiter++;
          break;
        case av1_obu::ObuType::SEQUENCE_HEADER:
          num_sequence_header++;
          if (!first_sequence_header_pos) {
            first_sequence_header_pos = obu->offset;
          }
          break;
        case av1_obu::ObuType::FRAME_HEADER:
        case av1_obu::ObuType::FRAME:
          num_frame++;
          if (!first_frame_pos) {
            first_frame_pos = obu->offset;
          }
          break;
        default:
          break;
      }
      pos += obu->size;
    }

    // The whole packet must parse as a series of OBUs (i.e. the IVF headers are stripped)
    EXPECT_EQ(pos, size);
    // Each packet must be exactly one temporal unit carrying exactly one shown frame
    // (AV1 spec Section 7.5)
    EXPECT_EQ(num_temporal_delimiter, 1U);
    EXPECT_EQ(num_frame, 1U);
    // The temporal unit must start with a temporal delimiter OBU (AV1 spec Section 7.5)
    ASSERT_TRUE(first_obu_type.has_value());
    EXPECT_EQ(*first_obu_type, av1_obu::ObuType::TEMPORAL_DELIMITER);

    ASSERT_TRUE(result.flags.has_value());
    if (result.flags.value() == 1) {
      // Key frame packets must be self-contained random access points: a single sequence
      // header OBU placed before the frame (AV1 spec Section 7.6.2)
      EXPECT_EQ(num_sequence_header, 1U);
      ASSERT_TRUE(first_sequence_header_pos.has_value());
      ASSERT_TRUE(first_frame_pos.has_value());
      // The sequence header must sit after the leading temporal delimiter and before the
      // first frame-bearing OBU, i.e. TD -> SH -> frame (AV1 spec Section 7.5). The spec
      // does not require SH to be immediately after TD (metadata OBUs may legally sit in
      // between), so the position is range-checked rather than compared for equality
      EXPECT_GE(*first_sequence_header_pos, first_obu_end);
      EXPECT_LT(*first_sequence_header_pos, *first_frame_pos);
    } else {
      EXPECT_EQ(num_sequence_header, 0U);
    }

    num_checked++;
  }
};

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

TEST_F(TestAV1RandomAccess, KeyFramesAreSelfContainedRandomAccessPoints)
{
  auto compressor = make_jetson_av1_compressor();
  constexpr int desired_i_frame_interval = 10;

  compressor->register_postprocess<
    TestAV1RandomAccess, &TestAV1RandomAccess::check_with_obu_layout<
                           common::ImageFormat::AV1, desired_i_frame_interval>>(this);

  for (auto & [name, value] : compressor->parameters()) {
    if (name == "i_frame_interval") {
      value = desired_i_frame_interval;
    }
  }

  for (auto i = 0; i < TestAV1RandomAccess::NUM_FRAMES; i++) {
    compressor->process(get_image());
  }

  // Wait until all fed frames are dequeued and checked while the compressor is fully alive.
  // Otherwise the tail frames are processed during destruction, where virtual dispatch
  // resolves payload_preprocess_impl / payload_copy_impl to the base class implementations
  // and raw IVF-wrapped packets reach the checker
  constexpr int expected_packets = TestAV1RandomAccess::NUM_FRAMES;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (num_checked.load() < expected_packets && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT_EQ(num_checked.load(), expected_packets);
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
