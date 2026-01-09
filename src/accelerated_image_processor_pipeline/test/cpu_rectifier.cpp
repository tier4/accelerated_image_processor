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

#include "accelerated_image_processor_pipeline/rectifier.hpp"
#include "test_utility.hpp"

#include <gtest/gtest.h>

namespace accelerated_image_processor::pipeline
{
TEST_F(TestRectifier, CpuRectificationDefault)
{
  auto rectifier = make_cpu_rectifier();
  rectifier->register_postprocess<TestRectifier, &TestRectifier::check>(this);
  rectifier->set_camera_info(get_camera_info());
  rectifier->process(get_image());
}

TEST_F(TestRectifier, CpuRectificationLowAlpha)
{
  auto rectifier = make_cpu_rectifier();
  rectifier->register_postprocess<TestRectifier, &TestRectifier::check>(this);
  for (auto & [name, value] : rectifier->parameters()) {
    if (name == "alpha") {
      value = 0.0;
    }
  }
  EXPECT_EQ(rectifier->parameter_value<double>("alpha"), 0.0);
  rectifier->set_camera_info(get_camera_info());
  rectifier->process(get_image());
}

TEST_F(TestRectifier, CpuRectificationHighAlpha)
{
  auto rectifier = make_cpu_rectifier();
  rectifier->register_postprocess<TestRectifier, &TestRectifier::check>(this);
  for (auto & [name, value] : rectifier->parameters()) {
    if (name == "alpha") {
      value = 1.0;
    }
  }
  EXPECT_EQ(rectifier->parameter_value<double>("alpha"), 1.0);
  rectifier->set_camera_info(get_camera_info());
  rectifier->process(get_image());
}
}  // namespace accelerated_image_processor::pipeline

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
