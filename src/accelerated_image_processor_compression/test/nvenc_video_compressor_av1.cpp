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

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <tuple>

#ifdef NVENC_AVAILABLE
namespace accelerated_image_processor::compression
{
namespace
{
/**
 * @brief Check the running machine exposes an NVENC AV1 encoder
 *
 * The encoder needs the NVIDIA display driver and a GPU whose NVENC engine supports AV1, hence
 * the tests below are skipped rather than failed when they are not available
 */
bool is_nvenc_av1_available()
{
  auto compressor = make_nvenc_av1_compressor();
  const auto [is_available, message] = compressor->validate_compression_type_compatibility();
  if (!is_available) {
    std::cerr << "NVENC AV1 encoder is not available: " << message << std::endl;
  }
  return is_available;
}

/**
 * @brief AV1 frame types (AV1 spec Section 6.8.2)
 */
enum class Av1FrameType : uint8_t { KEY = 0, INTER = 1, INTRA_ONLY = 2, SWITCH = 3 };

/**
 * @brief Read the frame type out of the uncompressed header of a frame bearing OBU
 *
 * The uncompressed header begins with show_existing_frame f(1), which is followed by
 * frame_type f(2) unless the frame just repeats an already decoded one (AV1 spec Section 5.9.2).
 *
 * @return The frame type, or std::nullopt when the OBU carries no frame type
 */
std::optional<Av1FrameType> read_frame_type(
  const uint8_t * data, const size_t size, const av1_obu::ObuRange & obu)
{
  const uint8_t obu_header = data[obu.offset];
  const size_t obu_header_size = 1 + ((obu_header & 0x04) != 0 ? 1 : 0);  // +1 if extended
  const auto obu_size = av1_obu::read_leb128(data, size, obu.offset + obu_header_size);
  if (!obu_size) {
    return std::nullopt;
  }

  const size_t payload_offset = obu.offset + obu_header_size + obu_size->length;
  if (payload_offset >= size) {
    return std::nullopt;
  }

  const uint8_t first_byte = data[payload_offset];
  if ((first_byte & 0x80) != 0) {
    return std::nullopt;  // show_existing_frame is set, hence no frame type follows
  }
  return static_cast<Av1FrameType>((first_byte >> 5) & 0x03);
}
}  // namespace

using AV1ParamCombination = std::tuple<
  bool /* av1.enable_tile */, int /* av1.log2_num_tile_row */, int /* av1.log2_num_tile_col */>;
using TestAV1Compressor = TestVideoCompressor<AV1ParamCombination>;

/**
 * @brief Fixture that additionally validates the OBU layout of every emitted AV1 packet
 */
class TestAV1RandomAccess : public TestAV1Compressor
{
public:
  int num_checked{0};

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
    std::optional<Av1FrameType> frame_type;

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
            frame_type = read_frame_type(data, size, *obu);
          }
          break;
        default:
          break;
      }
      pos += obu->size;
    }

    // The whole packet must parse as a series of OBUs
    EXPECT_EQ(pos, size);
    // Each packet must be exactly one temporal unit carrying exactly one shown frame
    // (AV1 spec Section 7.5)
    EXPECT_EQ(num_temporal_delimiter, 1U);
    EXPECT_EQ(num_frame, 1U);
    // The temporal unit must start with a temporal delimiter OBU (AV1 spec Section 7.5)
    ASSERT_TRUE(first_obu_type.has_value());
    EXPECT_EQ(*first_obu_type, av1_obu::ObuType::TEMPORAL_DELIMITER);

    ASSERT_TRUE(result.flags.has_value());
    ASSERT_TRUE(frame_type.has_value());
    if (result.flags.value() == 1) {
      // Key frame packets must be self-contained random access points: an actual key frame
      // (an INTRA_ONLY frame is NOT enough, since a decoder cannot start decoding at it)
      // preceded by a single sequence header OBU (AV1 spec Section 7.6.2)
      EXPECT_EQ(*frame_type, Av1FrameType::KEY);
      EXPECT_EQ(num_sequence_header, 1U);
      ASSERT_TRUE(first_sequence_header_pos.has_value());
      ASSERT_TRUE(first_frame_pos.has_value());
      // The sequence header must sit after the leading temporal delimiter and before the
      // first frame-bearing OBU, i.e. TD -> SH -> frame (AV1 spec Section 7.5)
      EXPECT_GE(*first_sequence_header_pos, first_obu_end);
      EXPECT_LT(*first_sequence_header_pos, *first_frame_pos);
    } else {
      EXPECT_EQ(*frame_type, Av1FrameType::INTER);
      EXPECT_EQ(num_sequence_header, 0U);
    }

    num_checked++;
  }
};

TEST_F(TestAV1Compressor, NvencVideoCompressorAV1Default)
{
  if (!is_nvenc_av1_available()) {
    GTEST_SKIP() << "NVENC AV1 encoder is not available on this machine";
  }

  auto compressor = make_nvenc_av1_compressor();
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
  if (!is_nvenc_av1_available()) {
    GTEST_SKIP() << "NVENC AV1 encoder is not available on this machine";
  }

  auto compressor = make_nvenc_av1_compressor();
  constexpr int desired_i_frame_interval = 10;

  compressor->register_postprocess<
    TestAV1RandomAccess, &TestAV1RandomAccess::check_with_obu_layout<
                           common::ImageFormat::AV1, desired_i_frame_interval>>(this);

  for (auto & [name, value] : compressor->parameters()) {
    if (name == "i_frame_interval" || name == "idr_frame_interval") {
      value = desired_i_frame_interval;
    }
  }

  for (auto i = 0; i < TestAV1RandomAccess::NUM_FRAMES; i++) {
    compressor->process(get_image());
  }

  // The compressor encodes synchronously, hence every fed frame must have been checked already
  EXPECT_EQ(num_checked, TestAV1RandomAccess::NUM_FRAMES);
}

/**
 * @brief The key frame interval is the smaller of `i_frame_interval` and `idr_frame_interval`,
 * and every key frame stays a self-contained random access point
 *
 * NVENC would otherwise emit AV1 `INTRA_ONLY` frames, which carry no sequence header and which
 * a decoder cannot start decoding at, for the intra frames between two IDR frames
 */
template <int IFrameInterval, int IdrFrameInterval>
void run_key_frame_interval_case(TestAV1RandomAccess * fixture)
{
  auto compressor = make_nvenc_av1_compressor();
  constexpr int expected_key_frame_interval =
    IFrameInterval < IdrFrameInterval ? IFrameInterval : IdrFrameInterval;

  compressor->register_postprocess<
    TestAV1RandomAccess, &TestAV1RandomAccess::check_with_obu_layout<
                           common::ImageFormat::AV1, expected_key_frame_interval>>(fixture);

  for (auto & [name, value] : compressor->parameters()) {
    if (name == "i_frame_interval") {
      value = IFrameInterval;
    } else if (name == "idr_frame_interval") {
      value = IdrFrameInterval;
    }
  }

  for (auto i = 0; i < TestAV1RandomAccess::NUM_FRAMES; i++) {
    compressor->process(fixture->get_image());
  }

  EXPECT_EQ(fixture->num_checked, TestAV1RandomAccess::NUM_FRAMES);
}

TEST_F(TestAV1RandomAccess, KeyFrameIntervalFollowsTheShorterIFrameInterval)
{
  if (!is_nvenc_av1_available()) {
    GTEST_SKIP() << "NVENC AV1 encoder is not available on this machine";
  }

  run_key_frame_interval_case<5, 20>(this);
}

TEST_F(TestAV1RandomAccess, KeyFrameIntervalFollowsTheShorterIdrFrameInterval)
{
  if (!is_nvenc_av1_available()) {
    GTEST_SKIP() << "NVENC AV1 encoder is not available on this machine";
  }

  run_key_frame_interval_case<10, 5>(this);
}

TEST_F(TestAV1Compressor, NvencVideoCompressorAV1RejectsLossless)
{
  if (!is_nvenc_av1_available()) {
    GTEST_SKIP() << "NVENC AV1 encoder is not available on this machine";
  }

  auto compressor = make_nvenc_av1_compressor();
  for (auto & [name, value] : compressor->parameters()) {
    if (name == "compression_type") {
      value = static_cast<std::string>("lossless");
    }
  }

  // NVENC does not expose lossless encoding for AV1, which has to be reported instead of being
  // silently encoded as a lossy stream
  const auto [is_valid, message] = compressor->validate_compression_type_compatibility();
  EXPECT_FALSE(is_valid);
  EXPECT_FALSE(message.empty());
  EXPECT_THROW(compressor->process(get_image()), std::runtime_error);
}

TEST_P(TestAV1Compressor, NvencVideoCompressorAV1TileTypeCombo)
{
  if (!is_nvenc_av1_available()) {
    GTEST_SKIP() << "NVENC AV1 encoder is not available on this machine";
  }

  auto [enable_tile, log2_num_tile_row, log2_num_tile_col] = GetParam();
  auto compressor = make_nvenc_av1_compressor();
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
    }
  }
  EXPECT_EQ(compressor->parameter_value<int>("i_frame_interval"), desired_i_frame_interval);
  EXPECT_EQ(compressor->parameter_value<bool>("av1.enable_tile"), enable_tile);
  EXPECT_EQ(compressor->parameter_value<int>("av1.log2_num_tile_row"), log2_num_tile_row);
  EXPECT_EQ(compressor->parameter_value<int>("av1.log2_num_tile_col"), log2_num_tile_col);

  for (auto i = 0; i < TestAV1Compressor::NUM_FRAMES; i++) {
    compressor->process(get_image());
  }
}

INSTANTIATE_TEST_SUITE_P(
  NvencVideoCompressorAV1ComboWithTiling, TestAV1Compressor,
  ::testing::Combine(
    // Enable tiling
    ::testing::Values(true),
    // log2 num tile row
    ::testing::Values(1, 2),
    // log2 num tile col
    ::testing::Values(1, 2)));

INSTANTIATE_TEST_SUITE_P(
  NvencVideoCompressorAV1ComboWithoutTiling, TestAV1Compressor,
  ::testing::Combine(
    // Disable tiling
    ::testing::Values(false),
    // log2 num tile row (ignored)
    ::testing::Values(0),
    // log2 num tile col (ignored)
    ::testing::Values(0)));

}  // namespace accelerated_image_processor::compression
#else
TEST(NvencVideoCompressorAV1Skip, NvencUnavailable)
{
  GTEST_SKIP() << "NVENC not available. Skipping NvencVideoCompressorAV1 tests.";
}
#endif
int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
