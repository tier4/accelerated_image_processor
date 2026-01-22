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

#pragma once

#include <accelerated_image_processor_common/processor.hpp>
#include <rclcpp/node.hpp>

#include <string>

namespace accelerated_image_processor::ros
{
/**
 * @brief Fetch parameters from ROS parameter server and set them to the processor.
 *
 * @param node The ROS node to fetch parameters from.
 * @param processor The processor to set the parameters to.
 * @param prefix The prefix of the parameter names.
 */
void fetch_parameters(
  rclcpp::Node * node, common::BaseProcessor * processor, const std::string & prefix);

/**
 * @brief Fetch parameters from ROS parameter server and set them to the processor.
 *
 * @param node The ROS node to fetch parameters from.
 * @param processor The processor to set the parameters to.
 */
inline void fetch_parameters(rclcpp::Node * node, common::BaseProcessor * processor)
{
  fetch_parameters(node, processor, "");
}
}  // namespace accelerated_image_processor::ros
