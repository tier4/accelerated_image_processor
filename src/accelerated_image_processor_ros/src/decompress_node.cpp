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

#include "decompress_node.hpp"

#include "accelerated_image_processor_decompression/builder.hpp"
#include "accelerated_image_processor_ros/conversion.hpp"
#include "accelerated_image_processor_ros/parameter.hpp"
#include "accelerated_image_processor_ros/qos.hpp"

#include <sensor_msgs/msg/detail/image__struct.hpp>

#include <chrono>
#include <memory>
#include <string>
#include <utility>

namespace accelerated_image_processor::ros
{
DecompressNode::DecompressNode(const rclcpp::NodeOptions & options)
: Node("decompression_node", options)
{
  auto max_task_length = this->declare_parameter<int>("max_task_length");

  qos_request_timer_ = rclcpp::create_timer(
    this, this->get_clock(), std::chrono::milliseconds(100),
    [this, max_task_length]() { this->determine_qos(max_task_length); });
}

void DecompressNode::determine_qos(const int max_task_length)
{
  auto compressed_topic =
    this->get_node_topics_interface()->resolve_topic_name("image_raw/compressed", false);

  rclcpp::QoS compressed_qos(1);
  constexpr int throttle_period_ms = 1000;
  if (!find_qos(this, compressed_topic, compressed_qos, throttle_period_ms)) {
    RCLCPP_ERROR_THROTTLE(
      this->get_logger(), *this->get_clock(), throttle_period_ms,
      "Failed to find image QoS settings");
    return;
  }

  std::string topic_type = "";
  if (!find_topic_type(this, compressed_topic, topic_type, throttle_period_ms)) {
    RCLCPP_ERROR_THROTTLE(
      this->get_logger(), *this->get_clock(), throttle_period_ms,
      "Failed to find compressed topic type");
    return;
  }

  if (topic_type == "ffmpeg_image_transport_msgs/msg/FFMPEGPacket") {
    compressed_subscription_ =
      this->create_subscription<ffmpeg_image_transport_msgs::msg::FFMPEGPacket>(
        compressed_topic, compressed_qos,
        [this](const ffmpeg_image_transport_msgs::msg::FFMPEGPacket::ConstSharedPtr msg) {
          this->on_ffmpeg_packet(msg);
        });

    // video decompressor
    {
      decompressor_ =
        decompression::create_decompressor<DecompressNode, &DecompressNode::publish_decompressed>(
          "video", this);
      fetch_parameters(this, decompressor_.get(), "decompressor");
    }
  } else {
    throw std::runtime_error("Unsupported compression type");
  }

  decompressed_publisher_ =
    this->create_publisher<sensor_msgs::msg::Image>("image_raw", compressed_qos);

  decompression_worker_.emplace(max_task_length);

  // once all queries received, stop the timer callback1
  qos_request_timer_->cancel();
}

void DecompressNode::on_ffmpeg_packet(
  const ffmpeg_image_transport_msgs::msg::FFMPEGPacket::ConstSharedPtr msg)
{
  const auto image = std::make_shared<const common::Image>(from_ros_ffmpeg(*msg));

  // image decompression
  if (decompression_worker_) {
    // NOTE: capture `msg` by value to extend the lifetime of the shared pointer at least until the
    // task is completed
    decompression_worker_->add_task([this, image, msg]() { decompressor_->process(*image); });
  }
}

void DecompressNode::publish_decompressed(const common::Image & image)
{
  auto decompressed = to_ros_raw(image);
  decompressed_publisher_->publish(std::move(decompressed));
}

}  // namespace accelerated_image_processor::ros

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(accelerated_image_processor::ros::DecompressNode)
