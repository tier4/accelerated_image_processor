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

#include <accelerated_image_processor_common/datatype.hpp>
#include <accelerated_image_processor_common/parameter.hpp>
#include <accelerated_image_processor_common/processor.hpp>

#include <memory>

namespace accelerated_image_processor::decompression
{
class FfmpegVideoDecompressor;

/**
 * @brief Enumeration of video decompression backends.
 */
enum class VideoBackend : uint8_t { FFMPEG };

/**
 * @brief Abstract base class for video decompressors.
 */
class VideoDecompressor : public common::BaseProcessor
{
public:
  explicit VideoDecompressor(VideoBackend backend, common::ParameterMap dedicated_parameters = {})
  : BaseProcessor(dedicated_parameters += {}), backend_(backend)
  {
  }

  ~VideoDecompressor() override = default;

  /**
   * @brief Return the backend enum
   */
  VideoBackend backend() const { return backend_; }

private:
  const VideoBackend backend_;  //!< Decompression backend type.
};
//!< @brief Factory function to create a FfmpegVideoDecompressor.
std::unique_ptr<VideoDecompressor> make_ffmpeg_video_decompressor();
}  // namespace accelerated_image_processor::decompression
