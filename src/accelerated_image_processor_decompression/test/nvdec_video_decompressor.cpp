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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

// Decompression headers
#include <accelerated_image_processor_common/datatype.hpp>
#include <accelerated_image_processor_decompression/builder.hpp>
#include <accelerated_image_processor_decompression/video_decompressor.hpp>

namespace accelerated_image_processor::decompression
{
namespace
{
std::size_t decoded_frame_count = 0;
std::vector<uint64_t> decoded_pts;

/**
 * @brief a free function to check the decoded result
 *
 * A single packet can produce multiple frames. The video decompressor extracts each frame one by
 * one and feeds it to postprocess().
 * This function is registered as a callback to check the every decoded frames.
 */
template <int WIDTH, int HEIGHT>
void check_nvdec_video_decompressor_result(const common::Image & decoded)
{
  ++decoded_frame_count;
  EXPECT_GT(decoded.data.size(), 0U);
  EXPECT_EQ(decoded.width, WIDTH);
  EXPECT_EQ(decoded.height, HEIGHT);
  EXPECT_EQ(decoded.step, WIDTH * 3U);
  EXPECT_EQ(decoded.data.size(), static_cast<std::size_t>(WIDTH) * HEIGHT * 3U);
  EXPECT_EQ(decoded.format, common::ImageFormat::RAW);
  EXPECT_EQ(decoded.encoding, common::ImageEncoding::BGR);
  EXPECT_TRUE(decoded.pts.has_value());
  if (decoded.pts) {
    decoded_pts.push_back(*decoded.pts);
  }

  // // This raw file can be converted to the bmp file like:
  // //   $convert -size 2880x1860 -depth 8 bgr:test_0.raw -colorspace RGB test_0.raw.bmp
  // std::ofstream outfile;
  // static int count = 0;
  // outfile.open("/tmp/test_" + std::to_string(count++) + ".raw", std::ios_base::app);
  // outfile.write(reinterpret_cast<const char *>(decoded.data.data()), decoded.data.size());
  // outfile.close();
}

template <typename CodecProvider>
void test_video_decompression()
{
  decoded_frame_count = 0;
  decoded_pts.clear();
  auto decompressor = make_nvdec_video_decompressor();
  ASSERT_NE(decompressor, nullptr);

  FfmpegTestDataProvider<CodecProvider> generator;

  auto checker = check_nvdec_video_decompressor_result<generator.WIDTH, generator.HEIGHT>;
  decompressor->register_postprocess(checker);

  while (std::optional<common::Image> frame = generator.next()) {
    decompressor->process(*frame);
  }

  common::Image end_of_stream{};
  end_of_stream.format = CodecProvider::format;
  decompressor->process(end_of_stream);

  EXPECT_EQ(decoded_frame_count, static_cast<std::size_t>(generator.NUM_FRAMES));
  ASSERT_EQ(decoded_pts.size(), static_cast<std::size_t>(generator.NUM_FRAMES));
  for (std::size_t index = 0; index < decoded_pts.size(); ++index) {
    EXPECT_EQ(decoded_pts[index], index);
  }
  EXPECT_FALSE(decompressor->exception_ptr());
}

TEST(TestNvDecVideoDecompressor, DecompressH264)
{
  test_video_decompression<H264Provider>();
}

TEST(TestNvDecVideoDecompressor, DecompressH265)
{
  test_video_decompression<H265Provider>();
}

TEST(TestNvDecVideoDecompressor, DecompressAV1)
{
  test_video_decompression<AV1Provider>();
}
}  // namespace
}  // namespace accelerated_image_processor::decompression

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
