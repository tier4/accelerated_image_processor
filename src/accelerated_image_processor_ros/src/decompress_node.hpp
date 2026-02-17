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

#pragma once

#include "accelerated_image_processor_ros/task_queue.hpp"

#include <accelerated_image_processor_common/datatype.hpp>
#include <accelerated_image_processor_decompression/builder.hpp>
#include <rclcpp/publisher_base.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/subscription_base.hpp>

#include <ffmpeg_image_transport_msgs/msg/ffmpeg_packet.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <memory>
#include <optional>

namespace accelerated_image_processor::ros
{
class DecompressNode : public rclcpp::Node
{
public:
  explicit DecompressNode(const rclcpp::NodeOptions & options);

private:
  /**
   * @brief Determine the QoS for the node based on the input parameters.
   * @param max_task_length Maximum task length.
   */
  void determine_qos(const int max_task_length);

  /**
   * @brief Callback function for video encoded messages.
   * @param msg Shared pointer to the received ffmpeg packet message.
   */
  void on_ffmpeg_packet(const ffmpeg_image_transport_msgs::msg::FFMPEGPacket::ConstSharedPtr msg);

  /**
   * @brief Callback function for publishing decompressed images.
   * @param image The decompressed image compressed and to be published.
   */
  void publish_decompressed(const common::Image & image);

  // --- image processors ---
  std::unique_ptr<decompression::Decompressor> decompressor_;

  // --- subscriptions and publishers ---
  rclcpp::SubscriptionBase::SharedPtr compressed_subscription_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr decompressed_publisher_;

  rclcpp::TimerBase::SharedPtr qos_request_timer_;

  std::optional<TaskWorker> decompression_worker_;

  bool use_jpeg_compression_;
};
}  // namespace accelerated_image_processor::ros
