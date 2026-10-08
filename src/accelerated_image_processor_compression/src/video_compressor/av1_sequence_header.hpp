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

#include "av1_obu.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

/**
 * Minimal reader/writer of color_config() in the AV1 sequence header OBU.
 *
 * Unlike av1_obu.hpp, which only locates OBU boundaries, this header interprets the bit fields
 * of the sequence header OBU payload (AV1 spec Section 5.5) so that the color description
 * (color primaries, transfer characteristics, matrix coefficients and color range) can be read
 * or replaced. This is needed for encoders that give no control over these fields (e.g. the
 * Jetson AV1 encoder), where the sequence header has to be patched to describe the actual input.
 *
 * color_config() sits near the end of the sequence header and is followed only by
 * film_grain_params_present and the trailing bits:
 *
 *   +---------------------------+----------------+-------------------------+---------------+
 *   | seq_profile ...           | color_config() | film_grain_params_      | trailing_bits |
 *   | enable_restoration        |                | present (1 bit)         |               |
 *   +---------------------------+----------------+-------------------------+---------------+
 *
 * Therefore rewriting it only needs to copy the leading bits verbatim, emit a new
 * color_config(), and re-emit the rest, whose length may change by 24 bits depending on
 * color_description_present_flag.
 */
namespace accelerated_image_processor::compression::av1_sequence_header
{
/**
 * @brief Color description carried by color_config() (AV1 spec Section 6.4.2)
 *
 * The values follow ISO/IEC 23091-4/ITU-T H.273, e.g. 1 stands for BT.709 and 6 for BT.601
 */
struct ColorConfig
{
  uint8_t color_primaries;
  uint8_t transfer_characteristics;
  uint8_t matrix_coefficients;
  bool color_range;  //!< false: studio swing (limited), true: full swing (full)

  bool operator==(const ColorConfig & other) const
  {
    return color_primaries == other.color_primaries &&
           transfer_characteristics == other.transfer_characteristics &&
           matrix_coefficients == other.matrix_coefficients && color_range == other.color_range;
  }
  bool operator!=(const ColorConfig & other) const { return !(*this == other); }
};

/**
 * @brief BT.709 primaries/transfer/matrix with the full range, which every video compressor of
 * this package feeds its encoder with
 */
inline constexpr ColorConfig bt709_full_range{1, 1, 1, true};

namespace detail
{
/**
 * @brief MSB-first bit reader (AV1 spec Section 4.10)
 *
 * read_bits(n) and read_uvlc() correspond to the descriptors f(n) and uvlc() the syntax tables of
 * the AV1 spec use. Reading past the end yields 0 and makes ok() false, so that a parser can check
 * the result once at the end rather than after every read.
 */
class BitReader
{
public:
  BitReader(const uint8_t * data, const size_t size) : data_(data), size_in_bit_(size * 8) {}

  //! Read n (<= 32) bits as an unsigned integer, i.e. f(n) (AV1 spec Section 4.10.2)
  uint32_t read_bits(const size_t n)
  {
    uint32_t value = 0;
    for (size_t i = 0; i < n; ++i) {
      if (pos_ >= size_in_bit_) {
        ok_ = false;
        return 0;
      }
      value = (value << 1) | ((data_[pos_ >> 3] >> (7 - (pos_ & 7))) & 1);
      ++pos_;
    }
    return value;
  }

  //! Read a variable length unsigned integer, i.e. uvlc() (AV1 spec Section 4.10.3)
  uint32_t read_uvlc()
  {
    size_t leading_zeros = 0;
    while (ok_ && read_bits(1) == 0) {
      ++leading_zeros;
    }
    if (!ok_ || leading_zeros >= 32) {
      return UINT32_MAX;
    }
    return read_bits(leading_zeros) + static_cast<uint32_t>((1ULL << leading_zeros) - 1);
  }

  bool bit_at(const size_t pos) const { return (data_[pos >> 3] >> (7 - (pos & 7))) & 1; }
  size_t position() const { return pos_; }
  bool ok() const { return ok_; }

private:
  const uint8_t * data_;
  size_t size_in_bit_;
  size_t pos_{0};
  bool ok_{true};
};

/**
 * @brief MSB-first bit writer
 */
class BitWriter
{
public:
  void write_bit(const bool bit)
  {
    if (pos_ % 8 == 0) {
      bytes_.push_back(0);
    }
    if (bit) {
      bytes_.back() |= static_cast<uint8_t>(0x80 >> (pos_ % 8));
    }
    ++pos_;
  }

  //! Write the lowest n bits of value, MSB first
  void write_bits(const uint32_t value, const size_t n)
  {
    for (size_t i = n; i > 0; --i) {
      write_bit(((value >> (i - 1)) & 1) != 0);
    }
  }

  //! trailing_bits(): a single 1 followed by 0s up to the byte boundary (AV1 spec Section 5.3.4)
  void write_trailing_bits()
  {
    write_bit(true);
    while (pos_ % 8 != 0) {
      write_bit(false);
    }
  }

  const std::vector<uint8_t> & bytes() const { return bytes_; }

private:
  std::vector<uint8_t> bytes_;
  size_t pos_{0};
};

/**
 * @brief Fields of the sequence header that rewriting color_config() has to know about
 */
struct SequenceHeaderLayout
{
  uint32_t seq_profile;
  //! Bit offset of color_config() from the head of the OBU payload
  size_t color_config_offset;
  bool high_bitdepth;
  bool mono_chrome;
  ColorConfig color;
  uint32_t chroma_sample_position;  //!< CSP_UNKNOWN (0) unless conveyed explicitly
  bool separate_uv_delta_q;
  bool film_grain_params_present;
};

/**
 * @brief Parse the sequence header OBU payload up to film_grain_params_present
 * (AV1 spec Section 5.5)
 *
 * @return std::nullopt when the payload is truncated or uses a profile other than 0 and 1
 */
inline std::optional<SequenceHeaderLayout> parse_sequence_header(
  const uint8_t * payload, const size_t size)
{
  BitReader reader(payload, size);
  SequenceHeaderLayout layout{};

  layout.seq_profile = reader.read_bits(3);
  if (layout.seq_profile > 1) {
    return std::nullopt;  // profile 2 (4:2:2 / 12 bit) is not used by this package
  }
  reader.read_bits(1);  // still_picture
  const bool reduced_still_picture_header = reader.read_bits(1);

  if (reduced_still_picture_header) {
    reader.read_bits(5);  // seq_level_idx[0]
  } else {
    bool decoder_model_info_present_flag = false;
    uint32_t buffer_delay_length = 0;
    if (reader.read_bits(1)) {  // timing_info_present_flag
      // timing_info()
      reader.read_bits(32);       // num_units_in_display_tick
      reader.read_bits(32);       // time_scale
      if (reader.read_bits(1)) {  // equal_picture_interval
        reader.read_uvlc();       // num_ticks_per_picture_minus_1
      }
      decoder_model_info_present_flag = reader.read_bits(1);
      if (decoder_model_info_present_flag) {
        // decoder_model_info()
        buffer_delay_length = reader.read_bits(5) + 1;  // buffer_delay_length_minus_1
        reader.read_bits(32);                           // num_units_in_decoding_tick
        reader.read_bits(5);                            // buffer_removal_time_length_minus_1
        reader.read_bits(5);                            // frame_presentation_time_length_minus_1
      }
    }
    const bool initial_display_delay_present_flag = reader.read_bits(1);
    const uint32_t operating_points_cnt = reader.read_bits(5) + 1;
    for (uint32_t i = 0; i < operating_points_cnt && reader.ok(); ++i) {
      reader.read_bits(12);  // operating_point_idc[i]
      // seq_level_idx[i], which is followed by seq_tier[i] for the levels above 3.3
      if (reader.read_bits(5) > 7) {
        reader.read_bits(1);  // seq_tier[i]
      }
      // decoder_model_present_for_this_op[i]
      if (decoder_model_info_present_flag && reader.read_bits(1)) {
        // operating_parameters_info(i)
        reader.read_bits(buffer_delay_length);  // decoder_buffer_delay[i]
        reader.read_bits(buffer_delay_length);  // encoder_buffer_delay[i]
        reader.read_bits(1);                    // low_delay_mode_flag[i]
      }
      // initial_display_delay_present_for_this_op[i]
      if (initial_display_delay_present_flag && reader.read_bits(1)) {
        reader.read_bits(4);  // initial_display_delay_minus_1[i]
      }
    }
  }

  const uint32_t frame_width_bits = reader.read_bits(4) + 1;
  const uint32_t frame_height_bits = reader.read_bits(4) + 1;
  reader.read_bits(frame_width_bits);   // max_frame_width_minus_1
  reader.read_bits(frame_height_bits);  // max_frame_height_minus_1
  // frame_id_numbers_present_flag
  if (!reduced_still_picture_header && reader.read_bits(1)) {
    reader.read_bits(4);  // delta_frame_id_length_minus_2
    reader.read_bits(3);  // additional_frame_id_length_minus_1
  }
  reader.read_bits(1);  // use_128x128_superblock
  reader.read_bits(1);  // enable_filter_intra
  reader.read_bits(1);  // enable_intra_edge_filter
  if (!reduced_still_picture_header) {
    reader.read_bits(1);  // enable_interintra_compound
    reader.read_bits(1);  // enable_masked_compound
    reader.read_bits(1);  // enable_warped_motion
    reader.read_bits(1);  // enable_dual_filter
    const bool enable_order_hint = reader.read_bits(1);
    if (enable_order_hint) {
      reader.read_bits(1);  // enable_jnt_comp
      reader.read_bits(1);  // enable_ref_frame_mvs
    }
    // In the spec, seq_force_screen_content_tools takes 0, 1 or SELECT_SCREEN_CONTENT_TOOLS (= 2):
    //
    //   seq_choose_screen_content_tools                      f(1)
    //   if (seq_choose_screen_content_tools)
    //     seq_force_screen_content_tools = SELECT_SCREEN_CONTENT_TOOLS
    //   else
    //     seq_force_screen_content_tools                     f(1)
    //   if (seq_force_screen_content_tools > 0) { ... }
    //
    // Since the sequence header only tests it against `> 0`, it is held as a bool here, which is
    // NOT a copy of seq_choose_screen_content_tools: it is true either when chosen per frame (2)
    // or when the following bit is 1. The short-circuit `||` reads that bit only when not chosen,
    // which matches the spec exactly
    const bool seq_choose_screen_content_tools = reader.read_bits(1);
    const bool seq_force_screen_content_tools =
      seq_choose_screen_content_tools || reader.read_bits(1);
    if (seq_force_screen_content_tools) {
      if (!reader.read_bits(1)) {  // seq_choose_integer_mv
        reader.read_bits(1);       // seq_force_integer_mv
      }
    }
    if (enable_order_hint) {
      reader.read_bits(3);  // order_hint_bits_minus_1
    }
  }
  reader.read_bits(1);  // enable_superres
  reader.read_bits(1);  // enable_cdef
  reader.read_bits(1);  // enable_restoration

  // color_config() (AV1 spec Section 5.5.2). Profile 0 and 1 never carry twelve_bit
  layout.color_config_offset = reader.position();
  layout.high_bitdepth = reader.read_bits(1);
  layout.mono_chrome = (layout.seq_profile == 1) ? false : reader.read_bits(1);
  if (reader.read_bits(1)) {  // color_description_present_flag
    layout.color.color_primaries = static_cast<uint8_t>(reader.read_bits(8));
    layout.color.transfer_characteristics = static_cast<uint8_t>(reader.read_bits(8));
    layout.color.matrix_coefficients = static_cast<uint8_t>(reader.read_bits(8));
  } else {
    layout.color.color_primaries = 2;           // CP_UNSPECIFIED
    layout.color.transfer_characteristics = 2;  // TC_UNSPECIFIED
    layout.color.matrix_coefficients = 2;       // MC_UNSPECIFIED
  }
  // CP_BT_709, TC_SRGB and MC_IDENTITY
  const bool is_srgb = layout.color.color_primaries == 1 &&
                       layout.color.transfer_characteristics == 13 &&
                       layout.color.matrix_coefficients == 0;
  if (!layout.mono_chrome && is_srgb) {
    layout.color.color_range = true;  // implied, 4:4:4 without chroma_sample_position
  } else {
    layout.color.color_range = reader.read_bits(1);
    // Profile 0 other than monochrome is 4:2:0, which conveys chroma_sample_position
    if (!layout.mono_chrome && layout.seq_profile == 0) {
      layout.chroma_sample_position = reader.read_bits(2);
    }
  }
  layout.separate_uv_delta_q = layout.mono_chrome ? false : reader.read_bits(1);

  layout.film_grain_params_present = reader.read_bits(1);

  if (!reader.ok()) {
    return std::nullopt;
  }
  return layout;
}

/**
 * @brief Byte range of the payload within a sequence header OBU
 */
struct PayloadRange
{
  size_t header_size;     //!< Size of the OBU header (and extension) in bytes
  size_t payload_offset;  //!< Offset of the payload from the OBU head
  size_t payload_size;    //!< Payload size in bytes, i.e. obu_size
};

/**
 * @brief Locate the payload of a sequence header OBU
 * @return std::nullopt when the data is not a sequence header OBU carrying obu_size
 */
inline std::optional<PayloadRange> sequence_header_payload(const uint8_t * obu, const size_t size)
{
  const auto range = av1_obu::next_obu(obu, size, 0);
  if (!range || range->type != av1_obu::ObuType::SEQUENCE_HEADER) {
    return std::nullopt;
  }
  if ((obu[0] & 0x02) == 0) {
    return std::nullopt;  // obu_has_size_field is required to locate the payload end
  }
  const size_t header_size = (obu[0] & 0x04) != 0 ? 2 : 1;
  const auto obu_size = av1_obu::read_leb128(obu, size, header_size);
  if (!obu_size) {
    return std::nullopt;
  }
  return PayloadRange{
    header_size, header_size + obu_size->length, static_cast<size_t>(obu_size->value)};
}
}  // namespace detail

/**
 * @brief Read the color description out of a sequence header OBU
 *
 * @param obu Head of the sequence header OBU (starting from its OBU header byte)
 * @param size Size of the OBU in bytes
 * @return std::nullopt when the data is not a parsable sequence header OBU of profile 0 or 1
 */
inline std::optional<ColorConfig> read_color_config(const uint8_t * obu, const size_t size)
{
  const auto payload = detail::sequence_header_payload(obu, size);
  if (!payload) {
    return std::nullopt;
  }
  const auto layout =
    detail::parse_sequence_header(obu + payload->payload_offset, payload->payload_size);
  if (!layout) {
    return std::nullopt;
  }
  return layout->color;
}

/**
 * @brief Build a copy of the sequence header OBU whose color_config() describes `color`
 *
 * Every field other than the color description is kept as is. The color description is always
 * written explicitly (color_description_present_flag = 1), hence the OBU may grow by 3 bytes,
 * and obu_size is updated accordingly.
 *
 * @param obu Head of the sequence header OBU (starting from its OBU header byte)
 * @param size Size of the OBU in bytes
 * @param color Color description to be written
 * @return The rewritten OBU, or std::nullopt when the OBU cannot be handled: unparsable,
 * profile 2, monochrome, lacking obu_size, or `color` being the sRGB combination
 * (BT.709/sRGB/identity), which AV1 reserves for 4:4:4 with the implied full range
 */
inline std::optional<std::vector<uint8_t>> rewrite_color_config(
  const uint8_t * obu, const size_t size, const ColorConfig & color)
{
  const auto payload_range = detail::sequence_header_payload(obu, size);
  if (!payload_range) {
    return std::nullopt;
  }
  const uint8_t * payload = obu + payload_range->payload_offset;
  const auto layout = detail::parse_sequence_header(payload, payload_range->payload_size);
  if (!layout || layout->mono_chrome) {
    return std::nullopt;
  }
  if (
    color.color_primaries == 1 && color.transfer_characteristics == 13 &&
    color.matrix_coefficients == 0) {
    return std::nullopt;
  }

  detail::BitWriter writer;
  // Bits before color_config() stay untouched
  const detail::BitReader reader(payload, payload_range->payload_size);
  for (size_t pos = 0; pos < layout->color_config_offset; ++pos) {
    writer.write_bit(reader.bit_at(pos));
  }

  // color_config()
  writer.write_bit(layout->high_bitdepth);
  if (layout->seq_profile != 1) {
    writer.write_bit(false);  // mono_chrome
  }
  writer.write_bit(true);  // color_description_present_flag
  writer.write_bits(color.color_primaries, 8);
  writer.write_bits(color.transfer_characteristics, 8);
  writer.write_bits(color.matrix_coefficients, 8);
  writer.write_bit(color.color_range);
  if (layout->seq_profile == 0) {
    writer.write_bits(layout->chroma_sample_position, 2);  // 4:2:0
  }
  writer.write_bit(layout->separate_uv_delta_q);

  writer.write_bit(layout->film_grain_params_present);
  writer.write_trailing_bits();

  // Reassemble the OBU: the header (and extension) byte(s), obu_size in leb128, then the payload
  const auto & new_payload = writer.bytes();
  std::vector<uint8_t> result(obu, obu + payload_range->header_size);
  uint64_t remaining = new_payload.size();
  do {
    uint8_t byte = remaining & 0x7F;
    remaining >>= 7;
    if (remaining != 0) {
      byte |= 0x80;
    }
    result.push_back(byte);
  } while (remaining != 0);
  result.insert(result.end(), new_payload.begin(), new_payload.end());
  return result;
}
}  // namespace accelerated_image_processor::compression::av1_sequence_header
