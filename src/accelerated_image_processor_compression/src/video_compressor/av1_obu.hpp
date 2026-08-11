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

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

/**
 * Minimal scanner for the AV1 low overhead bitstream format.
 *
 * An AV1 stream is a series of OBUs (Open Bitstream Units). Each OBU consists of:
 *
 *   +-----------------+----------------------+------------------+--------------+
 *   | obu_header      | obu_extension_header | obu_size         | payload      |
 *   | (1 byte)        | (1 byte, optional)   | (leb128, if      | (obu_size    |
 *   |                 |                      |  has_size_field) |  bytes)      |
 *   +-----------------+----------------------+------------------+--------------+
 *
 * where obu_header is laid out as (AV1 spec Section 5.3.2):
 *
 *   bit 7          : obu_forbidden_bit (must be 0)
 *   bits 6-3       : obu_type
 *   bit 2          : obu_extension_flag
 *   bit 1          : obu_has_size_field
 *   bit 0          : obu_reserved_1bit
 *
 * This scanner only needs to locate OBU boundaries and types; it never interprets OBU
 * payloads.
 */
namespace accelerated_image_processor::compression::av1_obu
{
/**
 * @brief OBU type values defined in the AV1 specification (Section 6.2.2)
 */
enum class ObuType : uint8_t {
  SEQUENCE_HEADER = 1,
  TEMPORAL_DELIMITER = 2,
  FRAME_HEADER = 3,
  TILE_GROUP = 4,
  METADATA = 5,
  FRAME = 6,
  REDUNDANT_FRAME_HEADER = 7,
  TILE_LIST = 8,
  PADDING = 15,
};

/**
 * @brief Byte range of a single OBU within a payload
 */
struct ObuRange
{
  size_t offset;  //!< Offset of the OBU header byte from the payload head
  size_t size;    //!< Total OBU size in bytes, including header/extension/size field bytes
  ObuType type;   //!< OBU type parsed from the header byte
};

/**
 * @brief Decoded leb128 value and its encoded length (AV1 spec Section 4.10.5)
 */
struct Leb128
{
  //! Decoded value. When used for obu_size, this is the size in bytes of the OBU payload
  //! (not including the header, extension, or the leb128 field itself)
  uint64_t value;
  //! Size in bytes that the variable-length leb128 encoding itself occupies (1 to 8)
  size_t length;
};

/**
 * @brief Read a leb128 encoded value starting at data[pos]
 * @return std::nullopt if the value is truncated or longer than the 8 byte maximum
 */
inline std::optional<Leb128> read_leb128(const uint8_t * data, const size_t size, const size_t pos)
{
  uint64_t value = 0;
  for (size_t i = 0; i < 8; ++i) {
    if (pos + i >= size) {
      return std::nullopt;  // Truncated
    }
    const uint8_t byte = data[pos + i];
    value |= static_cast<uint64_t>(byte & 0x7F) << (7 * i);
    if ((byte & 0x80) == 0) {
      return Leb128{value, i + 1};
    }
  }
  return std::nullopt;  // leb128 must not exceed 8 bytes
}

/**
 * @brief Parse the OBU starting at data[pos]
 *
 * If obu_has_size_field is 0, the OBU size is conveyed by other means (AV1 spec Section 5.2)
 * and the OBU is treated as extending to the end of the payload.
 *
 * @return std::nullopt when data[pos] cannot be a valid OBU (forbidden bit set, truncated
 * header or size field, or declared size exceeding the remaining payload)
 */
inline std::optional<ObuRange> next_obu(const uint8_t * data, const size_t size, const size_t pos)
{
  if (pos >= size) {
    return std::nullopt;
  }

  const uint8_t header = data[pos];
  if ((header & 0x80) != 0) {
    return std::nullopt;  // obu_forbidden_bit must be 0
  }
  const auto type = static_cast<ObuType>((header >> 3) & 0x0F);
  const bool has_extension = (header & 0x04) != 0;
  const bool has_size_field = (header & 0x02) != 0;
  const size_t header_size = has_extension ? 2 : 1;  // in byte
  if (pos + header_size > size) {
    return std::nullopt;  // Extension byte is truncated
  }
  if (!has_size_field) {
    return ObuRange{pos, size - pos, type};
  }
  const auto obu_size = read_leb128(data, size, pos + header_size);
  if (!obu_size) {
    return std::nullopt;
  }
  const size_t payload_begin = pos + header_size + obu_size->length;
  if (obu_size->value > size - payload_begin) {
    return std::nullopt;  // Declared size exceeds the remaining payload
  }
  return ObuRange{pos, header_size + obu_size->length + static_cast<size_t>(obu_size->value), type};
}

/**
 * @brief Find the first sequence header OBU in the payload
 */
inline std::optional<ObuRange> find_sequence_header(const uint8_t * data, const size_t size)
{
  size_t pos = 0;
  while (const auto obu = next_obu(data, size, pos)) {
    if (obu->type == ObuType::SEQUENCE_HEADER) {
      return obu;
    }
    pos += obu->size;
  }
  return std::nullopt;
}

/**
 * @brief Check whether the payload already contains a sequence header OBU
 */
inline bool contains_sequence_header(const uint8_t * data, const size_t size)
{
  return find_sequence_header(data, size).has_value();
}

/**
 * @brief Determine where a sequence header OBU should be inserted in a temporal unit
 *
 * AV1 spec Section 7.5 requires the sequence header to appear before the first frame header
 * OBU, while the temporal delimiter must stay at the very beginning of the temporal unit.
 * Therefore the insertion point is right after the leading temporal delimiter OBU(s), or the
 * payload head when no temporal delimiter is present (or the payload is unparsable).
 */
inline size_t sequence_header_insertion_offset(const uint8_t * data, const size_t size)
{
  size_t pos = 0;
  while (const auto obu = next_obu(data, size, pos)) {
    if (obu->type != ObuType::TEMPORAL_DELIMITER) {
      break;
    }
    pos += obu->size;
  }
  return pos;
}
}  // namespace accelerated_image_processor::compression::av1_obu
