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

#include "accelerated_image_processor_ros/parameter.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace accelerated_image_processor::ros
{
namespace
{
class DummyProcessor final : public common::BaseProcessor
{
public:
  explicit DummyProcessor(common::ParameterMap params) : BaseProcessor(std::move(params)) {}

private:
  common::Image process_impl(const common::Image & image) override { return image; }
};

std::shared_ptr<rclcpp::Node> make_node_with_overrides(
  const std::string & node_name, const std::vector<rclcpp::Parameter> & overrides)
{
  rclcpp::NodeOptions options;
  options.parameter_overrides(overrides);
  // Default in many distros is already "allow undeclared = false", but we keep options explicit.
  options.allow_undeclared_parameters(false);
  options.automatically_declare_parameters_from_overrides(false);

  return std::make_shared<rclcpp::Node>(node_name, options);
}
}  // namespace

TEST(TestParameterFetchParametersWithoutPrefix, NoOverridesUsesDefaults)
{
  auto node = make_node_with_overrides("test_fetch_defaults", {});

  DummyProcessor processor(
    common::ParameterMap{
      {"quality", 95},
      {"scale", 0.5},
      {"enabled", true},
      {"frame_id", std::string("camera")},
    });

  fetch_parameters(node.get(), &processor);

  // Parameters should be defaults
  EXPECT_EQ(processor.parameter_value<int>("quality"), 95);
  EXPECT_DOUBLE_EQ(processor.parameter_value<double>("scale"), 0.5);
  EXPECT_EQ(processor.parameter_value<bool>("enabled"), true);
  EXPECT_EQ(processor.parameter_value<std::string>("frame_id"), "camera");
}

TEST(TestParameterFetchParametersWithoutPrefix, AppliesOverrides)
{
  // overrides without prefix
  std::vector<rclcpp::Parameter> overrides{
    rclcpp::Parameter("quality", 10),
    rclcpp::Parameter("scale", 0.25),
    rclcpp::Parameter("enabled", false),
    rclcpp::Parameter("frame_id", std::string("overridden")),
  };

  auto node = make_node_with_overrides("test_fetch_overrides", overrides);

  DummyProcessor processor(
    common::ParameterMap{
      {"quality", 95},
      {"scale", 0.5},
      {"enabled", true},
      {"frame_id", std::string("camera")},
    });

  // Expect to fetch parameters successfully
  fetch_parameters(node.get(), &processor);

  // Parameters should be overridden
  EXPECT_EQ(processor.parameter_value<int>("quality"), 10);
  EXPECT_DOUBLE_EQ(processor.parameter_value<double>("scale"), 0.25);
  EXPECT_EQ(processor.parameter_value<bool>("enabled"), false);
  EXPECT_EQ(processor.parameter_value<std::string>("frame_id"), "overridden");
}

TEST(TestParameterFetchParametersWithPrefix, AppliesOverridesAndSuccessesToFetchWithCorrectPrefix)
{
  // overrides with "rectifier." prefix
  std::vector<rclcpp::Parameter> overrides{
    rclcpp::Parameter("rectifier.quality", 42),
    rclcpp::Parameter("rectifier.enabled", false),
  };

  auto node = make_node_with_overrides("test_fetch_prefix", overrides);

  DummyProcessor processor(
    common::ParameterMap{
      {"quality", 95},
      {"enabled", true},
    });

  // Expect to fetch parameters successfully with correct prefix
  fetch_parameters(node.get(), &processor, "rectifier");

  // Parameters should be overridden
  EXPECT_EQ(processor.parameter_value<int>("quality"), 42);
  EXPECT_EQ(processor.parameter_value<bool>("enabled"), false);
}

TEST(TestParameterFetchParametersWithPrefix, AppliesOverridesButFailsToFetchWithWrongPrefix)
{
  // overrides with "wrong." prefix
  std::vector<rclcpp::Parameter> overrides{
    rclcpp::Parameter("wrong.quality", 1),
  };
  auto node = make_node_with_overrides("test_wrong_prefix", overrides);

  DummyProcessor processor(
    common::ParameterMap{
      {"quality", 95},
      {"enabled", true},
    });

  // Expect to fetch parameters failed with wrong prefix
  fetch_parameters(node.get(), &processor, "rectifier");

  // Parameters should be defaults
  EXPECT_EQ(processor.parameter_value<int>("quality"), 95);
  EXPECT_EQ(processor.parameter_value<bool>("enabled"), true);
}
}  // namespace accelerated_image_processor::ros

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  rclcpp::init(argc, argv);

  const int rc = RUN_ALL_TESTS();

  rclcpp::shutdown();
  return rc;
}
