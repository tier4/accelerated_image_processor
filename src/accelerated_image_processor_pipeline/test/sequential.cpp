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

#include "accelerated_image_processor_pipeline/sequential.hpp"

#include "accelerated_image_processor_pipeline/rectifier.hpp"
#include "test_utility.hpp"

#include <accelerated_image_processor_compression/builder.hpp>

#include <gtest/gtest.h>

namespace accelerated_image_processor::pipeline
{
TEST(TestSequential, AppendWithoutFunction)
{
  Sequential sequential;
  sequential.append<Rectifier>("rectifier").append<compression::Compressor>("compressor", "jpeg");

  EXPECT_EQ(sequential.items().size(), 2);
}

TEST(TestSequential, AppendWithFreeFunction)
{
  Sequential sequential;
  sequential.append<Rectifier>("rectifier", &dummy_function)
    .append<compression::Compressor>("compressor", &dummy_function, "jpeg");

  EXPECT_EQ(sequential.items().size(), 2);
}

TEST(TestSequential, AppendWithMemberFunction)
{
  DummyClass dummy;

  Sequential sequential;
  sequential.append<Rectifier, DummyClass, &DummyClass::dummy_function>("rectifier", &dummy)
    .append<compression::Compressor, DummyClass, &DummyClass::dummy_function>(
      "compressor", &dummy, "jpeg");

  EXPECT_EQ(sequential.items().size(), 2);
}
}  // namespace accelerated_image_processor::pipeline
