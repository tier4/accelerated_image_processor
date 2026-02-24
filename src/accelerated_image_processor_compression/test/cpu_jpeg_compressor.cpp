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

#include "accelerated_image_processor_compression/jpeg_compressor.hpp"
#include "test_utility.hpp"

#include <gtest/gtest.h>

#ifdef TURBOJPEG_AVAILABLE
namespace accelerated_image_processor::compression
{
TEST_F(TestJPEGCompressor, CpuCompressionDefault)
{
  auto compressor = make_cpujpeg_compressor();
  compressor->register_postprocess<TestJPEGCompressor, &TestJPEGCompressor::check>(this);
  const auto result = compressor->process(get_image());
  EXPECT_TRUE(result);
}

TEST_F(TestJPEGCompressor, CpuCompressionWithLowQuality)
{
  auto compressor = make_cpujpeg_compressor();
  compressor->register_postprocess<TestJPEGCompressor, &TestJPEGCompressor::check>(this);
  for (auto & [name, value] : compressor->parameters()) {
    if (name == "quality") {
      value = 10;
    }
  }
  EXPECT_EQ(compressor->parameter_value<int>("quality"), 10);
  const auto result = compressor->process(get_image());
  EXPECT_TRUE(result);
}

TEST_F(TestJPEGCompressor, CpuCompressionWithHighQuality)
{
  auto compressor = make_cpujpeg_compressor();
  compressor->register_postprocess<TestJPEGCompressor, &TestJPEGCompressor::check>(this);
  for (auto & [name, value] : compressor->parameters()) {
    if (name == "quality") {
      value = 90;
    }
  }
  EXPECT_EQ(compressor->parameter_value<int>("quality"), 90);
  const auto result = compressor->process(get_image());
  EXPECT_TRUE(result);
}
}  // namespace accelerated_image_processor::compression
#else
TEST(CpuJPEGCompressorSkip, TurboJPEGUnavailable)
{
  GTEST_SKIP() << "TurboJPEG not available. Skipping CpuJPEGCompressor tests.";
}
#endif

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
