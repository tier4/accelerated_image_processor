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

#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>

namespace accelerated_image_processor::compression
{
#ifdef JETSON_AVAILABLE
/**
 * @brief H.265 encoder working on Jetson devices.
 */
class JetsonH265Compressor final : public JetsonVideoCompressor
{
public:
  inline static const std::unordered_map<std::string, v4l2_mpeg_video_h265_profile>
    h265_profile_map = {
      {"MAIN", V4L2_MPEG_VIDEO_H265_PROFILE_MAIN},
      {"MAIN10", V4L2_MPEG_VIDEO_H265_PROFILE_MAIN10},
    };

  inline static const std::unordered_map<std::string, v4l2_mpeg_video_h265_level> h265_level_map = {
    {"1_0_MAIN_TIER", V4L2_MPEG_VIDEO_H265_LEVEL_1_0_MAIN_TIER},
    {"1_0_HIGH_TIER", V4L2_MPEG_VIDEO_H265_LEVEL_1_0_HIGH_TIER},
    {"2_0_MAIN_TIER", V4L2_MPEG_VIDEO_H265_LEVEL_2_0_MAIN_TIER},
    {"2_0_HIGH_TIER", V4L2_MPEG_VIDEO_H265_LEVEL_2_0_HIGH_TIER},
    {"2_1_MAIN_TIER", V4L2_MPEG_VIDEO_H265_LEVEL_2_1_MAIN_TIER},
    {"2_1_HIGH_TIER", V4L2_MPEG_VIDEO_H265_LEVEL_2_1_HIGH_TIER},
    {"3_0_MAIN_TIER", V4L2_MPEG_VIDEO_H265_LEVEL_3_0_MAIN_TIER},
    {"3_0_HIGH_TIER", V4L2_MPEG_VIDEO_H265_LEVEL_3_0_HIGH_TIER},
    {"3_1_MAIN_TIER", V4L2_MPEG_VIDEO_H265_LEVEL_3_1_MAIN_TIER},
    {"3_1_HIGH_TIER", V4L2_MPEG_VIDEO_H265_LEVEL_3_1_HIGH_TIER},
    {"4_0_MAIN_TIER", V4L2_MPEG_VIDEO_H265_LEVEL_4_0_MAIN_TIER},
    {"4_0_HIGH_TIER", V4L2_MPEG_VIDEO_H265_LEVEL_4_0_HIGH_TIER},
    {"4_1_MAIN_TIER", V4L2_MPEG_VIDEO_H265_LEVEL_4_1_MAIN_TIER},
    {"4_1_HIGH_TIER", V4L2_MPEG_VIDEO_H265_LEVEL_4_1_HIGH_TIER},
    {"5_0_MAIN_TIER", V4L2_MPEG_VIDEO_H265_LEVEL_5_0_MAIN_TIER},
    {"5_0_HIGH_TIER", V4L2_MPEG_VIDEO_H265_LEVEL_5_0_HIGH_TIER},
    {"5_1_MAIN_TIER", V4L2_MPEG_VIDEO_H265_LEVEL_5_1_MAIN_TIER},
    {"5_1_HIGH_TIER", V4L2_MPEG_VIDEO_H265_LEVEL_5_1_HIGH_TIER},
    {"5_2_MAIN_TIER", V4L2_MPEG_VIDEO_H265_LEVEL_5_2_MAIN_TIER},
    {"5_2_HIGH_TIER", V4L2_MPEG_VIDEO_H265_LEVEL_5_2_HIGH_TIER},
    {"6_0_MAIN_TIER", V4L2_MPEG_VIDEO_H265_LEVEL_6_0_MAIN_TIER},
    {"6_0_HIGH_TIER", V4L2_MPEG_VIDEO_H265_LEVEL_6_0_HIGH_TIER},
    {"6_1_MAIN_TIER", V4L2_MPEG_VIDEO_H265_LEVEL_6_1_MAIN_TIER},
    {"6_1_HIGH_TIER", V4L2_MPEG_VIDEO_H265_LEVEL_6_1_HIGH_TIER},
    {"6_2_MAIN_TIER", V4L2_MPEG_VIDEO_H265_LEVEL_6_2_MAIN_TIER},
    {"6_2_HIGH_TIER", V4L2_MPEG_VIDEO_H265_LEVEL_6_2_HIGH_TIER},
  };

  JetsonH265Compressor()
  : JetsonVideoCompressor(
      SupportedCodec::H265, {{"h265.profile", static_cast<std::string>("MAIN")},
                             {"h265.level", static_cast<std::string>("5_1_MAIN_TIER")}})
  {
  }

  /**
   * @brief override impelmentation to validate the compatibility of the current compression
   * parameters.
   */
  std::tuple<bool, std::string> validate_compression_type_compatibility() override
  {
    // Load the latest parameters if the encoder is uninitialized; use the current config otherwise
    EncoderParameter latest_params;
    if (this->state_ == State::UNINITIALIZED) {
      this->collect_params(latest_params);
    } else {
      latest_params = this->encoder_params_;
    }

    this->collect_codec_params_impl(
      latest_params);  // this udpates class member `h264_profile_` and `h264_level_`

    auto ret = validate_compression_type_compatibility_impl(
      latest_params.compression_type, this->h265_profile_, this->h265_level_);
    return {ret.ok, ret.status.message};
  }

protected:
  /**
   * @brief class dedicated implementation to validate parameter compatibility
   */
  EncResult validate_compression_type_compatibility_impl(
    const VideoCompressionType & type, const v4l2_mpeg_video_h265_profile & profile,
    const v4l2_mpeg_video_h265_level & level)
  {
    // lossless compression requires high bitrate. lower level forces encoder work in bitrate that
    // is not enough for lossless compression, which may causes encoder initialization failure
    // and/or FD DMA mapping for capture plane. Though it is not explicitly documented, this class
    // treat the higher level is the only valid one for lossless.
    if (
      type == VideoCompressionType::LOSSLESS &&
      (level != V4L2_MPEG_VIDEO_H265_LEVEL_6_1_HIGH_TIER &&
       level != V4L2_MPEG_VIDEO_H265_LEVEL_6_2_HIGH_TIER)) {
      return EncResult(EncStatus(
        false,
        "Lossless compression is only supported with "
        "level 6_1_HIGH_TIER or 6_2_HIGH_TIER"));
    }

    return EncResult::success();
  }

  EncResult collect_codec_params_impl(
    [[maybe_unused]] const EncoderParameter & general_params) override
  {
    h265_profile_ = string_to_enum<v4l2_mpeg_video_h265_profile>(
      this->parameter_value<std::string>("h265.profile"), h265_profile_map);
    h265_level_ = string_to_enum<v4l2_mpeg_video_h265_level>(
      this->parameter_value<std::string>("h265.level"), h265_level_map);

    return EncResult::success();
  }

  EncResult set_capture_plane_format_impl(
    const uint32_t & width, const uint32_t & height, const uint32_t & image_size) override
  {
    CHECK_NVENC(
      encoder_->setCapturePlaneFormat(V4L2_PIX_FMT_H265, width, height, image_size),
      "Failed to set capture plane format to H265");

    return EncResult::success();
  }

  EncResult init_codec_impl() override
  {
    CHECK_NVENC(encoder_->setProfile(h265_profile_), "Failed to set H265 profile");

    CHECK_NVENC(encoder_->setLevel(h265_level_), "Failed to set H265 level");

    // Set chroma format and bit depth
    // This option is valid only for H.265
    uint8_t chroma_factor_idc =
      (encoder_params_.compression_type == VideoCompressionType::LOSSLESS) ? 3 : 1;
    CHECK_NVENC(
      encoder_->setChromaFactorIDC(chroma_factor_idc), "Failed to set H265 chroma factor IDC");

    // Insert Access Unit Delimiter (AUD) into encoded video stream
    // This insert Network Abstraction Layer (NAL) unit into the encoded
    // stream to explicitly show border of access unit (typically frame or picture)
    CHECK_NVENC(encoder_->setInsertAudEnabled(true), "Failed to  set inserting AUD for H265");

    // Insert Sequence Parameter Set (SPS) and Picture Parameter Set (PPS) to
    // each IDR frame so that decoders are able to decode stream from IDR frame surely
    CHECK_NVENC(
      encoder_->setInsertSpsPpsAtIdrEnabled(true), "Failed to set insertSPSPPSAtIDR for H265");

    return EncResult::success();
  }

private:
  v4l2_mpeg_video_h265_profile h265_profile_;
  v4l2_mpeg_video_h265_level h265_level_;
};

std::unique_ptr<VideoCompressor> make_jetson_h265_compressor()
{
  return std::make_unique<JetsonH265Compressor>();
}
#else
std::unique_ptr<VideoCompressor> make_jetson_h265_compressor()
{
  return nullptr;
}
#endif  // JETSON_AVAILABLE
}  // namespace accelerated_image_processor::compression
