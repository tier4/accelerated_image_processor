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

#include <rclcpp/exceptions/exceptions.hpp>

#include <string>
#include <type_traits>
#include <variant>

namespace accelerated_image_processor::ros
{
namespace
{
inline std::string format_prefix(const std::string & prefix, const std::string & ns)
{
  return prefix.empty() ? ns : prefix + "." + ns;
}
}  // namespace

void fetch_parameters(
  rclcpp::Node * node, common::BaseProcessor * processor, const std::string & prefix)
{
  for (auto & [name, value] : processor->parameters()) {
    const auto param_name = format_prefix(prefix, name);
    std::visit(
      [&](auto & v) {
        using T = std::decay_t<decltype(v)>;
        try {
          v = node->declare_parameter<T>(param_name, v);
        } catch (const rclcpp::exceptions::ParameterAlreadyDeclaredException & e) {
          v = node->get_parameter_or<T>(param_name, v);
        }
      },
      value);
  }
}

void fetch_parameters(
  rclcpp::Node * node, pipeline::Sequential & sequential, const std::string & prefix)
{
  for (auto & child : sequential.items()) {
    const auto child_prefix = format_prefix(prefix, child.ns);
    fetch_parameters(node, child.processor.get(), child_prefix);
  }
}
}  // namespace accelerated_image_processor::ros
