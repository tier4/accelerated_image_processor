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

#include "video_compressor/av1_obu.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace accelerated_image_processor::compression
{
namespace
{
// Synthetic OBUs. The OBU header byte is forbidden(1) | type(4) | extension(1) | has_size(1) |
// reserved(1)
const std::vector<uint8_t> temporal_delimiter = {0x12, 0x00};                 // type 2, size 0
const std::vector<uint8_t> sequence_header = {0x0A, 0x03, 0xAA, 0xBB, 0xCC};  // type 1, size 3
const std::vector<uint8_t> frame = {0x32, 0x02, 0xDE, 0xAD};                  // type 6, size 2

std::vector<uint8_t> concatenate(const std::vector<std::vector<uint8_t>> & parts)
{
  std::vector<uint8_t> result;
  for (const auto & part : parts) {
    result.insert(result.end(), part.begin(), part.end());
  }
  return result;
}
}  // namespace

TEST(Av1ObuReadLeb128, SingleByte)
{
  const std::vector<uint8_t> data = {0x05};
  const auto leb = av1_obu::read_leb128(data.data(), data.size(), 0);
  ASSERT_TRUE(leb.has_value());
  EXPECT_EQ(leb->value, 5U);
  EXPECT_EQ(leb->length, 1U);
}

TEST(Av1ObuReadLeb128, MultiByte)
{
  const std::vector<uint8_t> data = {0xE5, 0x8E, 0x26};
  const auto leb = av1_obu::read_leb128(data.data(), data.size(), 0);
  ASSERT_TRUE(leb.has_value());
  EXPECT_EQ(leb->value, 624485U);
  EXPECT_EQ(leb->length, 3U);
}

TEST(Av1ObuReadLeb128, Truncated)
{
  // The last byte still has the continuation bit set
  const std::vector<uint8_t> data = {0xE5, 0x8E};
  EXPECT_FALSE(av1_obu::read_leb128(data.data(), data.size(), 0).has_value());
}

TEST(Av1ObuReadLeb128, TooLong)
{
  // 8 continuation bytes exceed the maximum leb128 length
  const std::vector<uint8_t> data = {0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x01};
  EXPECT_FALSE(av1_obu::read_leb128(data.data(), data.size(), 0).has_value());
}

TEST(Av1ObuNextObu, ParsesSizedObus)
{
  const auto data = concatenate({temporal_delimiter, sequence_header, frame});

  const auto first = av1_obu::next_obu(data.data(), data.size(), 0);
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(first->offset, 0U);
  EXPECT_EQ(first->size, temporal_delimiter.size());
  EXPECT_EQ(first->type, av1_obu::ObuType::TEMPORAL_DELIMITER);

  const auto second = av1_obu::next_obu(data.data(), data.size(), first->size);
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(second->offset, temporal_delimiter.size());
  EXPECT_EQ(second->size, sequence_header.size());
  EXPECT_EQ(second->type, av1_obu::ObuType::SEQUENCE_HEADER);

  const auto third = av1_obu::next_obu(data.data(), data.size(), second->offset + second->size);
  ASSERT_TRUE(third.has_value());
  EXPECT_EQ(third->size, frame.size());
  EXPECT_EQ(third->type, av1_obu::ObuType::FRAME);

  // The three OBUs exactly cover the payload
  EXPECT_EQ(third->offset + third->size, data.size());
}

TEST(Av1ObuNextObu, HandlesExtensionByte)
{
  // Sequence header with obu_extension_flag set: header, extension, size, 2 payload bytes
  const std::vector<uint8_t> data = {0x0E, 0x00, 0x02, 0xAA, 0xBB};
  const auto obu = av1_obu::next_obu(data.data(), data.size(), 0);
  ASSERT_TRUE(obu.has_value());
  EXPECT_EQ(obu->size, data.size());
  EXPECT_EQ(obu->type, av1_obu::ObuType::SEQUENCE_HEADER);
}

TEST(Av1ObuNextObu, RejectsForbiddenBit)
{
  const std::vector<uint8_t> data = {0x80, 0x00};
  EXPECT_FALSE(av1_obu::next_obu(data.data(), data.size(), 0).has_value());
}

TEST(Av1ObuNextObu, RejectsTruncatedSizeField)
{
  // has_size_field is set but the buffer ends right after the header byte
  const std::vector<uint8_t> data = {0x0A};
  EXPECT_FALSE(av1_obu::next_obu(data.data(), data.size(), 0).has_value());
}

TEST(Av1ObuNextObu, RejectsSizeBeyondPayload)
{
  // Declared size (255) exceeds the remaining zero bytes
  const std::vector<uint8_t> data = {0x0A, 0xFF, 0x01};
  EXPECT_FALSE(av1_obu::next_obu(data.data(), data.size(), 0).has_value());
}

TEST(Av1ObuNextObu, WithoutSizeFieldExtendsToEnd)
{
  // Frame OBU with obu_has_size_field == 0
  const std::vector<uint8_t> data = {0x30, 0xDE, 0xAD};
  const auto obu = av1_obu::next_obu(data.data(), data.size(), 0);
  ASSERT_TRUE(obu.has_value());
  EXPECT_EQ(obu->size, data.size());
  EXPECT_EQ(obu->type, av1_obu::ObuType::FRAME);
}

TEST(Av1ObuFindSequenceHeader, FoundInTemporalUnit)
{
  const auto data = concatenate({temporal_delimiter, sequence_header, frame});
  const auto obu = av1_obu::find_sequence_header(data.data(), data.size());
  ASSERT_TRUE(obu.has_value());
  EXPECT_EQ(obu->offset, temporal_delimiter.size());
  EXPECT_EQ(obu->size, sequence_header.size());
  EXPECT_TRUE(av1_obu::contains_sequence_header(data.data(), data.size()));
}

TEST(Av1ObuFindSequenceHeader, Absent)
{
  const auto data = concatenate({temporal_delimiter, frame});
  EXPECT_FALSE(av1_obu::find_sequence_header(data.data(), data.size()).has_value());
  EXPECT_FALSE(av1_obu::contains_sequence_header(data.data(), data.size()));
}

TEST(Av1ObuInsertionOffset, AfterTemporalDelimiter)
{
  const auto data = concatenate({temporal_delimiter, frame});
  EXPECT_EQ(
    av1_obu::sequence_header_insertion_offset(data.data(), data.size()), temporal_delimiter.size());
}

TEST(Av1ObuInsertionOffset, AfterConsecutiveTemporalDelimiters)
{
  const auto data = concatenate({temporal_delimiter, temporal_delimiter, frame});
  EXPECT_EQ(
    av1_obu::sequence_header_insertion_offset(data.data(), data.size()),
    2 * temporal_delimiter.size());
}

TEST(Av1ObuInsertionOffset, ZeroWithoutTemporalDelimiter)
{
  const auto data = concatenate({frame});
  EXPECT_EQ(av1_obu::sequence_header_insertion_offset(data.data(), data.size()), 0U);
}

TEST(Av1ObuInsertionOffset, ZeroOnGarbage)
{
  const std::vector<uint8_t> data = {0x80, 0x12, 0x34};
  EXPECT_EQ(av1_obu::sequence_header_insertion_offset(data.data(), data.size()), 0U);
}

TEST(Av1ObuRoundTrip, KeyFrameBecomesRandomAccessPoint)
{
  // Emulate the compressor data path: extract the sequence header from the first payload and
  // insert it into a key frame payload with the same resize + 3x memcpy scheme as
  // JetsonAV1Compressor::payload_copy_impl
  const auto first_payload = concatenate({temporal_delimiter, sequence_header, frame});
  const auto key_payload = concatenate({temporal_delimiter, frame});

  const auto cached = av1_obu::find_sequence_header(first_payload.data(), first_payload.size());
  ASSERT_TRUE(cached.has_value());
  const std::vector<uint8_t> cache(
    first_payload.begin() + cached->offset, first_payload.begin() + cached->offset + cached->size);

  const size_t insert_pos =
    av1_obu::sequence_header_insertion_offset(key_payload.data(), key_payload.size());
  std::vector<uint8_t> result(key_payload.size() + cache.size());
  std::memcpy(result.data(), key_payload.data(), insert_pos);
  std::memcpy(result.data() + insert_pos, cache.data(), cache.size());
  std::memcpy(
    result.data() + insert_pos + cache.size(), key_payload.data() + insert_pos,
    key_payload.size() - insert_pos);

  // The result must be byte-identical to a temporal unit natively laid out as TD | SH | FRAME
  EXPECT_EQ(result, concatenate({temporal_delimiter, sequence_header, frame}));

  // Walk all OBUs: exactly one sequence header must appear, before the frame OBU, and the walk
  // must consume the whole payload
  size_t pos = 0;
  size_t num_sequence_header = 0;
  bool frame_seen = false;
  while (const auto obu = av1_obu::next_obu(result.data(), result.size(), pos)) {
    if (obu->type == av1_obu::ObuType::SEQUENCE_HEADER) {
      EXPECT_FALSE(frame_seen);  // The sequence header must precede the frame
      num_sequence_header++;
    } else if (obu->type == av1_obu::ObuType::FRAME) {
      frame_seen = true;
    }
    pos += obu->size;
  }
  EXPECT_EQ(pos, result.size());
  EXPECT_EQ(num_sequence_header, 1U);
  EXPECT_TRUE(frame_seen);
}
}  // namespace accelerated_image_processor::compression

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
