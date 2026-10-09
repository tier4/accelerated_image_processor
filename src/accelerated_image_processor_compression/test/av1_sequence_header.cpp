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

#include "video_compressor/av1_sequence_header.hpp"

#include "video_compressor/av1_obu.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace accelerated_image_processor::compression
{
namespace
{
using av1_sequence_header::ColorConfig;

// Sequence header OBUs emitted by real encoders, used as the ground truth. The color
// descriptions noted on the right were confirmed with ffprobe
//
// NVENC AV1 (RTX 4090), 640x480, 4:2:0
const std::vector<uint8_t> nvenc_bt601_full = {0x0A, 0x0E, 0x00, 0x00, 0x00, 0x24, 0xC4, 0xFF,
                                               0xDF, 0x00, 0x86, 0x64, 0x18, 0x18, 0x1A, 0x10};
// The same header carrying BT.709 / full range, which NVENC emits once configured so
const std::vector<uint8_t> nvenc_bt709_full = {0x0A, 0x0E, 0x00, 0x00, 0x00, 0x24, 0xC4, 0xFF,
                                               0xDF, 0x00, 0x86, 0x64, 0x04, 0x04, 0x06, 0x10};
// Hand-modified variants of the above: limited range, and no color description at all
const std::vector<uint8_t> nvenc_bt601_limited = {0x0A, 0x0E, 0x00, 0x00, 0x00, 0x24, 0xC4, 0xFF,
                                                  0xDF, 0x00, 0x86, 0x64, 0x18, 0x18, 0x18, 0x10};
const std::vector<uint8_t> nvenc_no_description = {0x0A, 0x0B, 0x00, 0x00, 0x00, 0x24, 0xC4,
                                                   0xFF, 0xDF, 0x00, 0x86, 0x60, 0x10};

// libaom (via ffmpeg), 64x48. The field layout before color_config() differs from NVENC's
const std::vector<uint8_t> aom_420_no_description = {0x0A, 0x0A, 0x00, 0x00, 0x00, 0x02,
                                                     0xAF, 0xF7, 0x9B, 0x5F, 0x20, 0x08};
const std::vector<uint8_t> aom_420_bt709_full = {0x0A, 0x0D, 0x00, 0x00, 0x00, 0x02, 0xAF, 0xF7,
                                                 0x9B, 0x5F, 0x22, 0x02, 0x02, 0x03, 0x08};
const std::vector<uint8_t> aom_420_bt601_full = {0x0A, 0x0D, 0x00, 0x00, 0x00, 0x02, 0xAF, 0xF7,
                                                 0x9B, 0x5F, 0x22, 0x0C, 0x0C, 0x0B, 0x08};
// seq_profile 1 (4:4:4), which has neither mono_chrome nor chroma_sample_position
const std::vector<uint8_t> aom_444_no_description = {0x0A, 0x0A, 0x20, 0x00, 0x00, 0x02,
                                                     0xAF, 0xF7, 0x9B, 0x5F, 0x20, 0x40};
const std::vector<uint8_t> aom_444_bt709_full = {0x0A, 0x0D, 0x20, 0x00, 0x00, 0x02, 0xAF, 0xF7,
                                                 0x9B, 0x5F, 0x24, 0x04, 0x04, 0x06, 0x40};
const std::vector<uint8_t> aom_monochrome = {0x0A, 0x0A, 0x00, 0x00, 0x00, 0x02,
                                             0xAF, 0xF7, 0x9B, 0x5F, 0x25, 0x40};

struct ReadCase
{
  std::string name;
  std::vector<uint8_t> obu;
  ColorConfig expected;
};

std::vector<ReadCase> read_cases()
{
  return {
    {"nvenc_bt601_full", nvenc_bt601_full, {6, 6, 6, true}},
    {"nvenc_bt709_full", nvenc_bt709_full, {1, 1, 1, true}},
    {"nvenc_bt601_limited", nvenc_bt601_limited, {6, 6, 6, false}},
    {"nvenc_no_description", nvenc_no_description, {2, 2, 2, false}},
    {"aom_420_no_description", aom_420_no_description, {2, 2, 2, false}},
    {"aom_420_bt709_full", aom_420_bt709_full, {1, 1, 1, true}},
    {"aom_420_bt601_full", aom_420_bt601_full, {6, 6, 5, true}},
    {"aom_444_no_description", aom_444_no_description, {2, 2, 2, false}},
    {"aom_444_bt709_full", aom_444_bt709_full, {1, 1, 1, true}},
    {"aom_monochrome", aom_monochrome, {2, 2, 2, true}},
  };
}
}  // namespace

TEST(Av1SequenceHeaderReadColorConfig, ReadsEncoderOutputs)
{
  for (const auto & [name, obu, expected] : read_cases()) {
    SCOPED_TRACE(name);
    const auto color = av1_sequence_header::read_color_config(obu.data(), obu.size());
    ASSERT_TRUE(color.has_value());
    EXPECT_EQ(color->color_primaries, expected.color_primaries);
    EXPECT_EQ(color->transfer_characteristics, expected.transfer_characteristics);
    EXPECT_EQ(color->matrix_coefficients, expected.matrix_coefficients);
    EXPECT_EQ(color->color_range, expected.color_range);
  }
}

TEST(Av1SequenceHeaderReadColorConfig, RejectsOtherObus)
{
  const std::vector<uint8_t> temporal_delimiter = {0x12, 0x00};
  EXPECT_FALSE(
    av1_sequence_header::read_color_config(temporal_delimiter.data(), temporal_delimiter.size())
      .has_value());
}

TEST(Av1SequenceHeaderReadColorConfig, RejectsTruncatedPayload)
{
  // Declare a payload that ends before color_config()
  std::vector<uint8_t> truncated(nvenc_bt709_full.begin(), nvenc_bt709_full.begin() + 6);
  truncated[1] = 4;  // obu_size
  EXPECT_FALSE(
    av1_sequence_header::read_color_config(truncated.data(), truncated.size()).has_value());
}

TEST(Av1SequenceHeaderRewriteColorConfig, MatchesEncoderOutputs)
{
  struct RewriteCase
  {
    std::string name;
    std::vector<uint8_t> source;
    std::vector<uint8_t> expected;
  };
  // The expected outputs are what the encoders themselves emit for BT.709 / full range, which
  // covers adding the color description (+3 bytes, obu_size update), flipping color_range, and
  // keeping the result unchanged when the description is already the requested one
  const std::vector<RewriteCase> cases = {
    {"nvenc_bt601_full", nvenc_bt601_full, nvenc_bt709_full},
    {"nvenc_bt601_limited", nvenc_bt601_limited, nvenc_bt709_full},
    {"nvenc_no_description", nvenc_no_description, nvenc_bt709_full},
    {"nvenc_bt709_full", nvenc_bt709_full, nvenc_bt709_full},
    {"aom_420_no_description", aom_420_no_description, aom_420_bt709_full},
    {"aom_420_bt601_full", aom_420_bt601_full, aom_420_bt709_full},
    {"aom_420_bt709_full", aom_420_bt709_full, aom_420_bt709_full},
    {"aom_444_no_description", aom_444_no_description, aom_444_bt709_full},
    {"aom_444_bt709_full", aom_444_bt709_full, aom_444_bt709_full},
  };

  for (const auto & [name, source, expected] : cases) {
    SCOPED_TRACE(name);
    const auto rewritten = av1_sequence_header::rewrite_color_config(
      source.data(), source.size(), av1_sequence_header::bt709_full_range);
    ASSERT_TRUE(rewritten.has_value());
    EXPECT_EQ(*rewritten, expected);

    // The result has to stay a single well-formed OBU
    const auto obu = av1_obu::next_obu(rewritten->data(), rewritten->size(), 0);
    ASSERT_TRUE(obu.has_value());
    EXPECT_EQ(obu->type, av1_obu::ObuType::SEQUENCE_HEADER);
    EXPECT_EQ(obu->size, rewritten->size());
  }
}

TEST(Av1SequenceHeaderRewriteColorConfig, WritesArbitraryDescription)
{
  const ColorConfig bt601_full{6, 6, 6, true};
  const auto rewritten = av1_sequence_header::rewrite_color_config(
    nvenc_bt709_full.data(), nvenc_bt709_full.size(), bt601_full);
  ASSERT_TRUE(rewritten.has_value());
  EXPECT_EQ(*rewritten, nvenc_bt601_full);
}

TEST(Av1SequenceHeaderRewriteColorConfig, RejectsUnsupportedInputs)
{
  const auto & color = av1_sequence_header::bt709_full_range;

  // Monochrome streams carry no chroma, hence describing them as BT.709 color is refused
  EXPECT_FALSE(
    av1_sequence_header::rewrite_color_config(aom_monochrome.data(), aom_monochrome.size(), color)
      .has_value());

  // Not a sequence header
  const std::vector<uint8_t> temporal_delimiter = {0x12, 0x00};
  EXPECT_FALSE(
    av1_sequence_header::rewrite_color_config(
      temporal_delimiter.data(), temporal_delimiter.size(), color)
      .has_value());

  // obu_has_size_field is cleared: the payload end cannot be told
  std::vector<uint8_t> no_size_field(nvenc_bt709_full);
  no_size_field[0] &= static_cast<uint8_t>(~0x02);
  no_size_field.erase(no_size_field.begin() + 1);
  EXPECT_FALSE(
    av1_sequence_header::rewrite_color_config(no_size_field.data(), no_size_field.size(), color)
      .has_value());

  // The sRGB combination is reserved for 4:4:4 with the implied full range
  const ColorConfig srgb{1, 13, 0, true};
  EXPECT_FALSE(
    av1_sequence_header::rewrite_color_config(
      nvenc_bt601_full.data(), nvenc_bt601_full.size(), srgb)
      .has_value());
}
}  // namespace accelerated_image_processor::compression
