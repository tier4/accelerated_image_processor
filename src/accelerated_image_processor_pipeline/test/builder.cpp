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

#include "accelerated_image_processor_pipeline/builder.hpp"

#include "accelerated_image_processor_pipeline/rectifier.hpp"

#include <gtest/gtest.h>

#include <memory>

namespace accelerated_image_processor::pipeline
{
#ifdef NPP_AVAILABLE
constexpr auto ExpectedBackend = RectifierBackend::NPP;
#elif OPENCV_CUDA_AVAILABLE
constexpr auto ExpectedBackend = RectifierBackend::OPENCV_CUDA;
#else
constexpr auto ExpectedBackend = RectifierBackend::CPU;
#endif

namespace
{
void check_rectifier_type(const std::unique_ptr<Rectifier> & rectifier)
{
  EXPECT_NE(rectifier, nullptr);

  auto ptr = dynamic_cast<Rectifier *>(rectifier.get());
  EXPECT_NE(ptr, nullptr);

  EXPECT_EQ(ptr->backend(), ExpectedBackend);
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

TEST(TestRectifierBuilder, CreateRectifier1)
{
  auto rectifier = create_rectifier();
  check_rectifier_type(rectifier);
}

TEST(TestRectifierBuilder, CreateRectifier2)
{
  auto rectifier = create_rectifier(&dummy_function);
  check_rectifier_type(rectifier);
}

TEST(TestRectifierBuilder, CreateRectifier3)
{
  DummyClass dummy;

  auto rectifier = create_rectifier<DummyClass, &DummyClass::dummy_function>(&dummy);
  check_rectifier_type(rectifier);
}
}  // namespace accelerated_image_processor::pipeline
