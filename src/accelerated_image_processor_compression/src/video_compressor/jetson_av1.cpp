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

#include "jetson.hpp"

#include <accelerated_image_processor_common/helper.hpp>

namespace accelerated_image_processor::compression
{
#ifdef JETSON_AVAILABLE

namespace
{
/**
 * @brief IVF header size (32 byte)
 */
constexpr size_t ivf_file_header_size_in_byte = 32;

/**
 * @brief IVF frame header size (12 byte)
 */
constexpr size_t ivf_frame_header_size_in_byte = 12;

/**
 * @brief IVF total header size (44 byte)
 */
constexpr size_t first_frame_header_size =
  ivf_file_header_size_in_byte + ivf_frame_header_size_in_byte;

}  // namespace

/**
 * @brief AV1 encoder working on Jetsonn devices.
 */
class JetsonAV1Compressor final : public JetsonVideoCompressor
{
public:
  /**
   * @brief constructor
   *
   * exposed parameters are:
   *   - enable_tile: if true, enable tiling division in AV1 codec, which leads parallel encoding
   *   - log2_num_tile_row: how many rows consisting of a tile in log2. ex. If 1 is given, 1 =
   * log2(2) -> 2 rows will be used. If 2 is given, 2 = log2(4) -> 4rows will be used
   *   - log2_num_tile_cols: how many columns consisting of a tile in log2. calculation is the same
   * as row's pattern
   *   - enable_ssim_rdo: flag if SSIM RDO (Variance based Structural Similarity Rate
   * Distortion Optimization) is enabled
   *   - enable_cdf_update: flag if CDF (Cumulative Distribution Function) is  enabled. If true, the
   * encoder updates CDF for entropy encoding every frame
   */
  JetsonAV1Compressor()
  : JetsonVideoCompressor(
      SupportedCodec::AV1, {{"av1.enable_tile", static_cast<bool>(true)},
                            {"av1.log2_num_tile_row", static_cast<int>(1)},
                            {"av1.log2_num_tile_col", static_cast<int>(1)},
                            {"av1.enable_ssim_rdo", static_cast<bool>(false)},
                            {"av1.enable_cdf_update", static_cast<bool>(true)}})
  {
  }

protected:
  /**
   * @brief Getter function to access private member
   */
  auto & header_cache() { return header_cache_; }

  EncResult collect_codec_params_impl() override
  {
    enable_tile_ = this->parameter_value<bool>("av1.enable_tile");
    log2_num_tile_row_ = this->parameter_value<int>("av1.log2_num_tile_row");
    log2_num_tile_col_ = this->parameter_value<int>("av1.log2_num_tile_col");
    enable_ssim_rdo_ = this->parameter_value<bool>("av1.enable_ssim_rdo");
    enable_cdf_update_ = this->parameter_value<bool>("av1.enable_cdf_update");

    return EncResult::success();
  }

  EncResult set_capture_plane_format_impl(
    const uint32_t & width, const uint32_t & height, const uint32_t & image_size) override
  {
    CHECK_NVENC(
      encoder_->setCapturePlaneFormat(V4L2_PIX_FMT_AV1, width, height, image_size),
      "Failed to set capture plane format to AV1");

    return EncResult::success();
  }

  EncResult init_codec_impl() override
  {
    if (enable_tile_) {
      v4l2_enc_av1_tile_config tile_config;
      tile_config.bEnableTile = enable_tile_;
      tile_config.nLog2RowTiles = log2_num_tile_row_;
      tile_config.nLog2ColTiles = log2_num_tile_col_;
      CHECK_NVENC(encoder_->enableAV1Tile(tile_config), "Failedd to enable AV1 tile configuration");
    }

    CHECK_NVENC(
      encoder_->setAV1SsimRdo(enable_ssim_rdo_),
      "Failed to set AV1's SSIM RDO (variance based Structural SImilarity Rate Distortion "
      "Optimization)");

    CHECK_NVENC(
      encoder_->setAV1DisableCDFUpdate(!enable_cdf_update_),
      "Failed to configure CDF update for AV1");

    return EncResult::success();
  }

  PayloadInfo payload_preprocess_impl(
    DqCallbackArgs * callback_args, const size_t bytes_used, const NvBuffer * buffer) override
  {
    auto [payload_ptr, payload_size, offset] =
      JetsonVideoCompressor::payload_preprocess_impl(callback_args, bytes_used, buffer);

    // Because jetson AV1 encoder wrap AV1 payload by IVF header, which leads to the decoder unable
    // to recognize AV1 payload, peel unnecessary header from the payload
    if (
      payload_size > 4 && payload_ptr[0] == 'D' && payload_ptr[1] == 'K' && payload_ptr[2] == 'I' &&
      payload_ptr[3] == 'F') {
      // IVF file header starts with 'DKIF'
      // Initial frame: skip IVF file header (32byte) + IVF frame header (12 bytes)
      offset = first_frame_header_size;
    } else {
      // 2nd frame and later: skip IVF frame header
      offset = ivf_frame_header_size_in_byte;
    }

    if (offset > payload_size) {
      throw std::runtime_error("Invalid AV1 payload observed");
    }

    payload_size = payload_size - offset;
    payload_ptr = payload_ptr + offset;

    // Caching the AV1 header information so that payload can be decoded without the very first
    // frame
    auto & header_cache = dynamic_cast<JetsonAV1Compressor *>(callback_args->obj)->header_cache();
    if (header_cache.empty() && offset == first_frame_header_size) {
      // Save whole payload including sequence header
      header_cache.resize(payload_size);
      std::memcpy(header_cache.data(), payload_ptr, payload_size);
    }

    return {payload_ptr, payload_size, offset};
  }

  void payload_copy_impl(
    const bool is_keyframe, const PayloadInfo & payload_info, DqCallbackArgs * callback_args,
    std::vector<uint8_t> & copy_destination) override
  {
    auto & header_cache = dynamic_cast<JetsonAV1Compressor *>(callback_args->obj)->header_cache();
    auto & [payload_ptr, payload_size, offset] = payload_info;

    if (is_keyframe && !header_cache.empty() && offset != first_frame_header_size) {
      // For the key frames, copy the AV1 sequence header so that decoders can start the process
      // from any key frames
      size_t header_size = header_cache.size();
      copy_destination.resize(header_size + payload_size);

      std::memcpy(copy_destination.data(), header_cache.data(), header_size);
      std::memcpy(copy_destination.data() + header_size, payload_ptr, payload_size);
    } else {
      JetsonVideoCompressor::payload_copy_impl(
        is_keyframe, payload_info, callback_args, copy_destination);
    }
  }

private:
  bool enable_tile_;
  int log2_num_tile_row_;
  int log2_num_tile_col_;
  bool enable_ssim_rdo_;
  bool enable_cdf_update_;
  std::vector<uint8_t> header_cache_;
};

std::unique_ptr<VideoCompressor> make_jetson_av1_compressor()
{
  return std::make_unique<JetsonAV1Compressor>();
}
#else
std::unique_ptr<VideoCompressor> make_jetson_av1_compressor()
{
  return nullptr;
}
#endif  // JETSON_AVAILABLE
}  // namespace accelerated_image_processor::compression
