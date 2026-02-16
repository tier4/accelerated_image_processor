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
/**
 * @brief H.264 encoder working on Jetsonn devices.
 */
class JetsonH264Compressor final : public JetsonVideoCompressor
{
public:
  inline static const std::unordered_map<std::string, v4l2_mpeg_video_h264_profile>
    h264_profile_map = {
      {"BASELINE", V4L2_MPEG_VIDEO_H264_PROFILE_BASELINE},
      {"MAIN", V4L2_MPEG_VIDEO_H264_PROFILE_MAIN},
      {"HIGH", V4L2_MPEG_VIDEO_H264_PROFILE_HIGH},
    };

  inline static const std::unordered_map<std::string, v4l2_mpeg_video_h264_level> h264_level_map = {
    {"1_0", V4L2_MPEG_VIDEO_H264_LEVEL_1_0}, {"1B", V4L2_MPEG_VIDEO_H264_LEVEL_1B},
    {"1_1", V4L2_MPEG_VIDEO_H264_LEVEL_1_1}, {"1_2", V4L2_MPEG_VIDEO_H264_LEVEL_1_2},
    {"1_3", V4L2_MPEG_VIDEO_H264_LEVEL_1_3}, {"2_0", V4L2_MPEG_VIDEO_H264_LEVEL_2_0},
    {"2_1", V4L2_MPEG_VIDEO_H264_LEVEL_2_1}, {"2_2", V4L2_MPEG_VIDEO_H264_LEVEL_2_2},
    {"3_0", V4L2_MPEG_VIDEO_H264_LEVEL_3_0}, {"3_1", V4L2_MPEG_VIDEO_H264_LEVEL_3_1},
    {"3_2", V4L2_MPEG_VIDEO_H264_LEVEL_3_2}, {"4_0", V4L2_MPEG_VIDEO_H264_LEVEL_4_0},
    {"4_1", V4L2_MPEG_VIDEO_H264_LEVEL_4_1}, {"4_2", V4L2_MPEG_VIDEO_H264_LEVEL_4_2},
    {"5_0", V4L2_MPEG_VIDEO_H264_LEVEL_5_0}, {"5_1", V4L2_MPEG_VIDEO_H264_LEVEL_5_1},
  };

  JetsonH264Compressor()
  : JetsonVideoCompressor(
      SupportedCodec::H264, {{"h264.profile", static_cast<std::string>("HIGH")},
                             {"h264.level", static_cast<std::string>("5_1")},
                             {"h264.enable_cabac", static_cast<bool>(true)}})
  {
  }

protected:
  EncResult collect_codec_params_impl() override
  {
    h264_profile_ = string_to_enum<v4l2_mpeg_video_h264_profile>(
      this->parameter_value<std::string>("h264.profile"), h264_profile_map);
    h264_level_ = string_to_enum<v4l2_mpeg_video_h264_level>(
      this->parameter_value<std::string>("h264.level"), h264_level_map);
    enable_cabac_ = this->parameter_value<bool>("h264.enable_cabac");

    return EncResult::success();
  }

  EncResult set_capture_plane_format_impl(
    const uint32_t & width, const uint32_t & height, const uint32_t & image_size) override
  {
    CHECK_NVENC(
      encoder_->setCapturePlaneFormat(V4L2_PIX_FMT_H264, width, height, image_size),
      "Failed to set capture plane format to H264");

    return EncResult::success();
  }

  EncResult init_codec_impl() override
  {
    CHECK_NVENC(encoder_->setProfile(h264_profile_), "Failed to set H264 profile");

    CHECK_NVENC(encoder_->setLevel(h264_level_), "Failed to set H264 level");

    // Enable/disable Context-Adaptive Binary Arithmetic Coding (CABAC)
    // This option is valid only for H.264
    CHECK_NVENC(encoder_->setCABAC(enable_cabac_), "Failed to set H264 CABAC");

    // Insert Access Unit Delimiter (AUD) into encoded video stream
    // This insert Network Abstraction Layer (NAL) unit into the encoded
    // stream to explicitly show border of access unit (typically frame or picture)
    CHECK_NVENC(encoder_->setInsertAudEnabled(true), "Failed to  set inserting AUD for H264");

    // Insert Sequence Parameter Set (SPS) and Picture Parameter Set (PPS) to
    // each IDR frame so that decoders are able to decode stream from IDR frame surely
    CHECK_NVENC(
      encoder_->setInsertSpsPpsAtIdrEnabled(true), "Failed to set insertSPSPPSAtIDR for H264");

    return EncResult::success();
  }

private:
  v4l2_mpeg_video_h264_profile h264_profile_;
  v4l2_mpeg_video_h264_level h264_level_;
  bool enable_cabac_;
};

std::unique_ptr<VideoCompressor> make_jetson_h264_compressor()
{
  return std::make_unique<JetsonH264Compressor>();
}
#else
std::unique_ptr<VideoCompressor> make_jetson_h264_compressor()
{
  return nullptr;
}
#endif  // JETSON_AVAILABLE
}  // namespace accelerated_image_processor::compression
