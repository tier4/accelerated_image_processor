// Copyright 2025 TIER IV, Inc.
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

#include "accelerated_image_processor_compression/builder.hpp"

#include "accelerated_image_processor_compression/jpeg_compressor.hpp"
#include "accelerated_image_processor_compression/video_compressor.hpp"

#include <accelerated_image_processor_common/datatype.hpp>

#include <gtest/gtest.h>

#include <memory>

namespace accelerated_image_processor::compression
{
#ifdef JETSON_AVAILABLE
constexpr auto ExpectedJPEGBackend = JPEGBackend::JETSON;
constexpr auto ExpectedVideoBackend = VideoBackend::JETSON;
#elif NVJPEG_AVAILABLE
constexpr auto ExpectedJPEGBackend = JPEGBackend::NVJPEG;
#else
constexpr auto ExpectedJPEGBackend = JPEGBackend::CPU;
#endif

namespace
{
/**
 * @brief Check compressor type by dynamic_cast.
 */
void check_compressor_type(const std::unique_ptr<Compressor> & compressor)
{
  EXPECT_NE(compressor, nullptr);

  auto ptr = dynamic_cast<JPEGCompressor *>(compressor.get());
  EXPECT_NE(ptr, nullptr);

  EXPECT_EQ(ptr->backend(), ExpectedJPEGBackend);
}

/**
 * @brief Check compressor type by dynamic_cast (for video encode).
 */
void check_video_compressor_type(const std::unique_ptr<Compressor> & compressor)
{
  EXPECT_NE(compressor, nullptr);

  auto ptr = dynamic_cast<VideoCompressor *>(compressor.get());
  EXPECT_NE(ptr, nullptr);

  EXPECT_EQ(ptr->backend(), ExpectedVideoBackend);
}

/**
 * @brief Dummy class to register postprocess function.
 */
struct DummyClass
{
  /**
   * @brief Dummy free function for postprocess.
   */
  void dummy_function(const common::Image &) {}
};

/**
 * @brief Dummy free function for postprocess.
 */
void dummy_function(const common::Image &)
{
}
}  // namespace

TEST(TestCompressorBuilder, CreateJPEGCompressor1)
{
  auto compressor = create_compressor(CompressionType::JPEG);
  check_compressor_type(compressor);
}

TEST(TestCompressorBuilder, CreateJPEGCompressor2)
{
  auto compressor = create_compressor("jpeg");
  check_compressor_type(compressor);
}

TEST(TestCompressorBuilder, CreateJPEGCompressor3)
{
  DummyClass dummy;

  auto compressor =
    create_compressor<DummyClass, &DummyClass::dummy_function>(CompressionType::JPEG, &dummy);
  check_compressor_type(compressor);
}

TEST(TestCompressorBuilder, CreateJPEGCompressor4)
{
  auto compressor = create_compressor(CompressionType::JPEG, &dummy_function);
  check_compressor_type(compressor);
}

TEST(TestCompressorBuilder, CreateJPEGCompressor5)
{
  DummyClass dummy;

  auto compressor = create_compressor<DummyClass, &DummyClass::dummy_function>("jpeg", &dummy);
  check_compressor_type(compressor);
}

TEST(TestCompressorBuilder, CreateJPEGCompressor6)
{
  auto compressor = create_compressor("jpeg", &dummy_function);
  check_compressor_type(compressor);
}

TEST(TestCompressorBuilder, CreateH264Compressor1)
{
  auto compressor = create_compressor(CompressionType::H264);
  check_video_compressor_type(compressor);
}

TEST(TestCompressorBuilder, CreateH264Compressor2)
{
  auto compressor = create_compressor("h264");
  check_video_compressor_type(compressor);
}

TEST(TestCompressorBuilder, CreateH264Compressor3)
{
  DummyClass dummy;

  auto compressor =
    create_compressor<DummyClass, &DummyClass::dummy_function>(CompressionType::H264, &dummy);
  check_video_compressor_type(compressor);
}

TEST(TestCompressorBuilder, CreateH264Compressor4)
{
  auto compressor = create_compressor(CompressionType::H264, &dummy_function);
  check_video_compressor_type(compressor);
}

TEST(TestCompressorBuilder, CreateH264Compressor5)
{
  DummyClass dummy;

  auto compressor = create_compressor<DummyClass, &DummyClass::dummy_function>("h264", &dummy);
  check_video_compressor_type(compressor);
}

TEST(TestCompressorBuilder, CreateH264Compressor6)
{
  auto compressor = create_compressor("h264", &dummy_function);
  check_video_compressor_type(compressor);
}

TEST(TestCompressorBuilder, CreateH265Compressor1)
{
  auto compressor = create_compressor(CompressionType::H265);
  check_video_compressor_type(compressor);
}

TEST(TestCompressorBuilder, CreateH265Compressor2)
{
  auto compressor = create_compressor("h265");
  check_video_compressor_type(compressor);
}

TEST(TestCompressorBuilder, CreateH265Compressor3)
{
  DummyClass dummy;

  auto compressor =
    create_compressor<DummyClass, &DummyClass::dummy_function>(CompressionType::H265, &dummy);
  check_video_compressor_type(compressor);
}

TEST(TestCompressorBuilder, CreateH265Compressor4)
{
  auto compressor = create_compressor(CompressionType::H265, &dummy_function);
  check_video_compressor_type(compressor);
}

TEST(TestCompressorBuilder, CreateH265Compressor5)
{
  DummyClass dummy;

  auto compressor = create_compressor<DummyClass, &DummyClass::dummy_function>("h265", &dummy);
  check_video_compressor_type(compressor);
}

TEST(TestCompressorBuilder, CreateH265Compressor6)
{
  auto compressor = create_compressor("h265", &dummy_function);
  check_video_compressor_type(compressor);
}

TEST(TestCompressorBuilder, CreateAV1Compressor1)
{
  auto compressor = create_compressor(CompressionType::AV1);
  check_video_compressor_type(compressor);
}

TEST(TestCompressorBuilder, CreateAV1Compressor2)
{
  auto compressor = create_compressor("av1");
  check_video_compressor_type(compressor);
}

TEST(TestCompressorBuilder, CreateAV1Compressor3)
{
  DummyClass dummy;

  auto compressor =
    create_compressor<DummyClass, &DummyClass::dummy_function>(CompressionType::AV1, &dummy);
  check_video_compressor_type(compressor);
}

TEST(TestCompressorBuilder, CreateAV1Compressor4)
{
  auto compressor = create_compressor(CompressionType::AV1, &dummy_function);
  check_video_compressor_type(compressor);
}

TEST(TestCompressorBuilder, CreateAV1Compressor5)
{
  DummyClass dummy;

  auto compressor = create_compressor<DummyClass, &DummyClass::dummy_function>("av1", &dummy);
  check_video_compressor_type(compressor);
}

TEST(TestCompressorBuilder, CreateAV1Compressor6)
{
  auto compressor = create_compressor("av1", &dummy_function);
  check_video_compressor_type(compressor);
}
}  // namespace accelerated_image_processor::compression
