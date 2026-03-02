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

#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>

// Decompression headers
#include <accelerated_image_processor_common/parameter.hpp>
#include <accelerated_image_processor_decompression/builder.hpp>
#include <accelerated_image_processor_decompression/video_decompressor.hpp>

namespace accelerated_image_processor::decompression
{

namespace
{
constexpr auto ExpectedBackend = VideoBackend::FFMPEG;

/**
 * @brief Check decompressor type by dynamic_cast.
 */
void check_decompressor_type(const std::unique_ptr<Decompressor> & decompressor)
{
  EXPECT_NE(decompressor, nullptr);

  auto ptr = dynamic_cast<VideoDecompressor *>(decompressor.get());
  EXPECT_NE(ptr, nullptr);

  EXPECT_EQ(ptr->backend(), ExpectedBackend);
}

/**
 * @brief Dummy class used to test the templated create_decompressor overload
 */
class DummyClass
{
public:
  void dummy_method(const common::Image & /*img*/) {}
};

/**
 * @brief Dummy free function for postprocess.
 */
void dummy_function(const common::Image &)
{
}
}  // namespace

TEST(TestDecompressorBuilder, CreateFFMPEGDecompressor1)
{
  auto decompressor = create_decompressor(DecompressionType::VIDEO);
  check_decompressor_type(decompressor);
}

TEST(TestDecompressorBuilder, CreateFFMPEGDecompressor2)
{
  auto decompressor = create_decompressor("video");
  check_decompressor_type(decompressor);
}

TEST(TestDecompressorBuilder, CreateFFMPEGDecompressor3)
{
  DummyClass dummy;

  auto decompressor =
    create_decompressor<DummyClass, &DummyClass::dummy_method>(DecompressionType::VIDEO, &dummy);
  check_decompressor_type(decompressor);
}

TEST(TestDecompressorBuilder, CreateFFMPEGDecompressor4)
{
  auto decompressor = create_decompressor(DecompressionType::VIDEO, &dummy_function);
  check_decompressor_type(decompressor);
}

TEST(TestDecompressorBuilder, CreateFFMPEGDecompressor5)
{
  DummyClass dummy;

  auto decompressor = create_decompressor<DummyClass, &DummyClass::dummy_method>("video", &dummy);
  check_decompressor_type(decompressor);
}

TEST(TestDecompressorBuilder, CreateFFMPEGDecompressor6)
{
  auto decompressor = create_decompressor("video", &dummy_function);
  check_decompressor_type(decompressor);
}

}  // namespace accelerated_image_processor::decompression

// Main function for the test executable
int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
