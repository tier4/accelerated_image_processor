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

#include "test_utility.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <ios>
#include <memory>
#include <optional>

// Decompression headers
#include <accelerated_image_processor_common/datatype.hpp>
#include <accelerated_image_processor_decompression/builder.hpp>
#include <accelerated_image_processor_decompression/video_decompressor.hpp>

namespace accelerated_image_processor::decompression
{
/**
 * @brief a free function to check the decoded result
 *
 * In FFmpeg, a single packet can contain multiple frames. The ffmpeg_video_decompressor
 * extracts each frame one by one and feeds it to postprocess().
 * This function is registered as a callback to check the every decoded frames.
 */
template <int WIDTH, int HEIGHT>
void check_ffmpeg_video_decompressor_result(const common::Image & decoded)
{
  EXPECT_GT(decoded.data.size(), 0U);
  EXPECT_EQ(decoded.width, WIDTH);
  EXPECT_EQ(decoded.height, HEIGHT);

  // // This raw file can be converted to the bmp file like:
  // //   $convert -size 2880x1860 -depth 8 bgr:test_0.raw -colorspace RGB test_0.raw.bmp
  // std::ofstream outfile;
  // static int count = 0;
  // outfile.open("/tmp/test_" + std::to_string(count++) + ".raw", std::ios_base::app);
  // outfile.write(reinterpret_cast<const char *>(decoded.data.data()), decoded.data.size());
  // outfile.close();
}

TEST(TestFFMPEGVideoDecompressor, DecompressH264)
{
  auto decompressor = make_ffmpeg_video_decompressor();
  ASSERT_NE(decompressor, nullptr);

  FfmpegTestDataProvider<H264Provider> generator;

  auto checker = check_ffmpeg_video_decompressor_result<generator.WIDTH, generator.HEIGHT>;
  decompressor->register_postprocess(checker);

  while (std::optional<common::Image> frame = generator.next()) {
    decompressor->process(*frame);
  }
}

TEST(TestFFMPEGVideoDecompressor, DecompressH265)
{
  auto decompressor = make_ffmpeg_video_decompressor();
  ASSERT_NE(decompressor, nullptr);

  FfmpegTestDataProvider<H265Provider> generator;

  auto checker = check_ffmpeg_video_decompressor_result<generator.WIDTH, generator.HEIGHT>;
  decompressor->register_postprocess(checker);

  while (std::optional<common::Image> frame = generator.next()) {
    decompressor->process(*frame);
  }
}

TEST(TestFFMPEGVideoDecompressor, DecompressAV1)
{
  auto decompressor = make_ffmpeg_video_decompressor();
  ASSERT_NE(decompressor, nullptr);

  FfmpegTestDataProvider<AV1Provider> generator;

  auto checker = check_ffmpeg_video_decompressor_result<generator.WIDTH, generator.HEIGHT>;
  decompressor->register_postprocess(checker);

  while (std::optional<common::Image> frame = generator.next()) {
    decompressor->process(*frame);
  }
}
}  // namespace accelerated_image_processor::decompression

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
