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

#include <rclcpp/rclcpp.hpp>

#include <optional>
#include <string>

namespace accelerated_image_processor::ros
{
/**
 * @brief Try to find the QoS profile for a given topic.
 *
 * @param node The ROS node to use for querying the topic.
 * @param topic_name The name of the topic to query.
 * @param throttle_period_ms The period in milliseconds to throttle the logging.
 * @return std::optional<rclcpp::QoS> The QoS profile for the topic, or std::nullopt if no
 * publishers are found or if multiple publishers are found.
 */
std::optional<rclcpp::QoS> find_qos(
  rclcpp::Node * node, const std::string & topic_name, int throttle_period_ms = 1000);

/**
 * @brief Try to find the QoS profile for a given topic.
 *
 * @param node The ROS node to use for querying the topic.
 * @param topic_name The name of the topic to query.
 * @param qos The QoS profile for the topic.
 * @return bool True if the QoS profile was found, false otherwise.
 */
bool find_qos(
  rclcpp::Node * node, const std::string & topic_name, rclcpp::QoS & qos,
  int throttle_period_ms = 1000);

/**
 * @brief Try to find the topic type for a given topic.
 *
 * @param node The ROS node to use for querying the topic.
 * @param topic_name The name of the topic to query.
 * @param throttle_period_ms The period in milliseconds to throttle the logging.
 * @return std::string The type for the topic, or an empty string if no
 * publishers are found or if multiple publishers are found.
 */
std::string find_topic_type(
  rclcpp::Node * node, const std::string & topic_name, int throttle_period_ms = 1000);

/**
 * @brief Try to find the topic type for a given topic.
 *
 * @param node The ROS node to use for querying the topic.
 * @param topic_name The name of the topic to query.
 * @param topic_type The type for the topic.
 * @return bool True if the topic type was found, false otherwise.
 */
bool find_topic_type(
  rclcpp::Node * node, const std::string & topic_name, std::string & topic_type,
  int throttle_period_ms = 1000);
}  // namespace accelerated_image_processor::ros
