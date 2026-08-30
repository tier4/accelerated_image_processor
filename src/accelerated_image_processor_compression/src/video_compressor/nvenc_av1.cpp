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

#include "nvenc.hpp"

#include <accelerated_image_processor_common/helper.hpp>

#include <algorithm>
#include <memory>
#include <string>

namespace accelerated_image_processor::compression
{
#ifdef NVENC_AVAILABLE

/**
 * @brief AV1 encoder working on a discrete NVIDIA GPU through NvEncodeAPI (NVENC).
 *
 * Unlike the Jetson hardware encoder, NVENC emits a plain OBU stream (no IVF container) where
 * every packet is a single temporal unit, and it inserts the sequence header OBU into every IDR
 * frame on its own. Therefore no payload rewriting is needed at all, as long as every key frame
 * the compressor reports is an IDR frame.
 *
 * That condition is not automatic: when the IDR period is longer than the GOP length, NVENC
 * emits intra frames in between that are AV1 `INTRA_ONLY` frames carrying no sequence header.
 * Since a decoder cannot start decoding at an `INTRA_ONLY` frame (AV1 spec Section 7.6.2 admits
 * a key frame accompanied by a sequence header only), such a frame must never be advertised as
 * a key frame. Hence this class drives both the GOP length and the IDR period with the same
 * value, see init_codec_impl().
 */
class NvencAV1Compressor final : public NvencVideoCompressor
{
public:
  /**
   * @brief constructor
   *
   * exposed parameters are:
   *   - av1.enable_tile: if true, enable tiling division in AV1 codec, which leads parallel
   * encoding
   *   - av1.log2_num_tile_row: how many rows consisting of a tile in log2. ex. If 1 is given, 1 =
   * log2(2) -> 2 rows will be used. If 2 is given, 2 = log2(4) -> 4rows will be used
   *   - av1.log2_num_tile_col: how many columns consisting of a tile in log2. calculation is the
   * same as row's pattern
   */
  NvencAV1Compressor()
  : NvencVideoCompressor(
      NvencSupportedCodec::AV1, {{"av1.enable_tile", static_cast<bool>(true)},
                                 {"av1.log2_num_tile_row", static_cast<int>(1)},
                                 {"av1.log2_num_tile_col", static_cast<int>(1)}})
  {
  }

  ~NvencAV1Compressor() override = default;

protected:
  GUID codec_guid() const override { return NV_ENC_CODEC_AV1_GUID; }

  EncResult collect_codec_params_impl(const EncoderParameter & general_params) override
  {
    enable_tile_ = this->parameter_value<bool>("av1.enable_tile");
    log2_num_tile_row_ = this->parameter_value<int>("av1.log2_num_tile_row");
    log2_num_tile_col_ = this->parameter_value<int>("av1.log2_num_tile_col");

    // Every key frame has to be an IDR frame (see the class documentation), hence the tighter of
    // the two intervals governs both of them
    key_frame_interval_ = std::min(general_params.i_frame_interval, general_params.idr_interval);

    if (enable_tile_) {
      if (log2_num_tile_row_ < 0 || log2_num_tile_col_ < 0) {
        return EncResult(
          EncStatus(false, "av1.log2_num_tile_row/col must be equal to or larger than 0"));
      }
      if (
        (1 << log2_num_tile_row_) > NV_MAX_TILE_ROWS_AV1 ||
        (1 << log2_num_tile_col_) > NV_MAX_TILE_COLS_AV1) {
        return EncResult(EncStatus(
          false, "The number of AV1 tile rows/columns must not exceed " +
                   std::to_string(NV_MAX_TILE_ROWS_AV1)));
      }
    }

    return EncResult::success();
  }

  EncResult init_codec_impl(NV_ENC_CONFIG & encode_config) override
  {
    auto & av1_config = encode_config.encodeCodecConfig.av1Config;

    // Emit an AV1 key frame, that is an IDR frame, every `key_frame_interval_` frames.
    //
    // The GOP length is narrowed down to the same value on purpose: an intra frame that is not
    // an IDR frame is encoded as an AV1 `INTRA_ONLY` frame, which carries no sequence header and
    // which a decoder cannot start decoding at. Driving both with one value keeps the guarantee
    // that every packet flagged as a key frame is a self-contained random access point. Note
    // that AV1 needs no equivalent of the H264/H265 "I frame that is not an IDR frame" because
    // its key frame already resets the reference frames
    encode_config.gopLength = static_cast<uint32_t>(key_frame_interval_);
    av1_config.idrPeriod = static_cast<uint32_t>(key_frame_interval_);

    // Output the sequence header for every IDR frame so that decoders can start decoding there
    av1_config.repeatSeqHdr = 1;
    av1_config.disableSeqHdr = 0;

    // Emit OBUs with their size field (not the Annex B format), which is what
    // ffmpeg_image_transport compatible decoders expect
    av1_config.outputAnnexBFormat = 0;

    // The encoder input is planar YUV 4:2:0, 8 bit
    av1_config.chromaFormatIDC = 1;
    av1_config.inputBitDepth = NV_ENC_BIT_DEPTH_8;
    av1_config.outputBitDepth = NV_ENC_BIT_DEPTH_8;

    // Color description has to be embedded to let decoders reproduce the source colors. The
    // input is converted to full range BT.601 YCbCr (see NvencVideoCompressor::convert_to_yuv)
    av1_config.colorPrimaries = NV_ENC_VUI_COLOR_PRIMARIES_SMPTE170M;
    av1_config.transferCharacteristics = NV_ENC_VUI_TRANSFER_CHARACTERISTIC_SMPTE170M;
    av1_config.matrixCoefficients = NV_ENC_VUI_MATRIX_COEFFS_SMPTE170M;
    av1_config.colorRange = 1;  // 0: limited (studio swing), 1: full

    // Tiling divides a frame into independently encodable regions, which the hardware encodes in
    // parallel. `enableCustomTileConfig = 0` lets the encoder divide the frame uniformly into the
    // requested number of tiles
    av1_config.enableCustomTileConfig = 0;
    if (enable_tile_) {
      av1_config.numTileRows = static_cast<uint32_t>(1 << log2_num_tile_row_);
      av1_config.numTileColumns = static_cast<uint32_t>(1 << log2_num_tile_col_);
    } else {
      av1_config.numTileRows = 1;
      av1_config.numTileColumns = 1;
    }

    return EncResult::success();
  }

private:
  bool enable_tile_{true};
  int log2_num_tile_row_{1};
  int log2_num_tile_col_{1};
  //! Interval of the key frames, which are always emitted as IDR frames
  int key_frame_interval_{10};
};

std::unique_ptr<VideoCompressor> make_nvenc_av1_compressor()
{
  return std::make_unique<NvencAV1Compressor>();
}
#else
std::unique_ptr<VideoCompressor> make_nvenc_av1_compressor()
{
  return nullptr;
}
#endif  // NVENC_AVAILABLE
}  // namespace accelerated_image_processor::compression
