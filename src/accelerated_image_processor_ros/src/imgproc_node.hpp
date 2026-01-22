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

#include "accelerated_image_processor_ros/task_queue.hpp"

#include <accelerated_image_processor_common/datatype.hpp>
#include <accelerated_image_processor_compression/builder.hpp>
#include <accelerated_image_processor_pipeline/builder.hpp>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <memory>
#include <optional>

namespace accelerated_image_processor::ros
{
class ImgProcNode : public rclcpp::Node
{
public:
  explicit ImgProcNode(const rclcpp::NodeOptions & options);

private:
  /**
   * @brief Determine the QoS for the node based on the input parameters.
   * @param do_rectify Whether rectification is enabled.
   * @param max_task_length Maximum task length.
   */
  void determine_qos(const bool do_rectify, const int max_task_length);

  /**
   * @brief Callback function for image messages.
   * @param msg Shared pointer to the received image message.
   */
  void on_image(const sensor_msgs::msg::Image::ConstSharedPtr msg);

  /**
   * @brief Callback function for camera info messages. This callback is called only once.
   * @param msg Shared pointer to the received camera info message.
   */
  void on_camera_info(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg);

  /**
   * @brief Callback function for publishing compressed images.
   * @param image The image to be compressed and published.
   */
  void publish_compressed(const common::Image & image);

  /**
   * @brief Callback function for publishing rectified raw images and camera info.
   * @param image The image to be rectified and published.
   */
  void publish_rectified_raw(const common::Image & image);

  /**
   * @brief Callback function for publishing rectified compressed images.
   * @param image The image to be rectified and compressed and published.
   */
  void publish_rectified_compressed(const common::Image & image);

  // --- image processors ---
  std::unique_ptr<compression::Compressor> raw_compressor_;

  std::unique_ptr<pipeline::Rectifier> raw_rectifier_;
  std::unique_ptr<compression::Compressor> rectified_compressor_;

  // --- subscriptions and publishers ---
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_subscription_;

  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_publisher_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rectified_raw_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr rectified_compressed_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr rectified_info_publisher_;

  rclcpp::TimerBase::SharedPtr qos_request_timer_;

  std::optional<TaskWorker> compression_worker_;
  std::optional<TaskWorker> rectification_worker_;
};
}  // namespace accelerated_image_processor::ros
