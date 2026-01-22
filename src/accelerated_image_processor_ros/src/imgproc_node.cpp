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

#include "imgproc_node.hpp"

#include "accelerated_image_processor_ros/conversion.hpp"
#include "accelerated_image_processor_ros/parameter.hpp"
#include "accelerated_image_processor_ros/qos.hpp"

#include <chrono>
#include <memory>
#include <string>
#include <utility>

namespace accelerated_image_processor::ros
{
ImgProcNode::ImgProcNode(const rclcpp::NodeOptions & options) : Node("imgproc_node", options)
{
  auto max_task_length = this->declare_parameter<int>("max_task_length");
  auto compression_type = this->declare_parameter<std::string>("compressor.type");
  auto do_rectify = this->declare_parameter<bool>("rectifier.do_rectify");

  // raw compressor
  {
    raw_compressor_ = compression::create_compressor<ImgProcNode, &ImgProcNode::publish_compressed>(
      compression_type, this);
    fetch_parameters(this, raw_compressor_.get(), "compressor");
  }

  // rectifier (rectification & compression)
  if (do_rectify) {
    // TODO(ktro2828):
    // - Implement composable processor that embraces multiple processors
    // - Enable to share a CUDA stream across multiple processors
    raw_rectifier_ =
      pipeline::create_rectifier<ImgProcNode, &ImgProcNode::publish_rectified_raw>(this);
    rectified_compressor_ =
      compression::create_compressor<ImgProcNode, &ImgProcNode::publish_rectified_compressed>(
        compression_type, this);

    fetch_parameters(this, raw_rectifier_.get(), "rectifier");
    fetch_parameters(this, rectified_compressor_.get(), "compressor");
  }

  qos_request_timer_ = rclcpp::create_timer(
    this, this->get_clock(), std::chrono::milliseconds(100),
    [this, do_rectify, max_task_length]() { this->determine_qos(do_rectify, max_task_length); });
}

void ImgProcNode::determine_qos(const bool do_rectify, const int max_task_length)
{
  auto image_topic = this->get_node_topics_interface()->resolve_topic_name("image_raw", false);
  auto info_topic = this->get_node_topics_interface()->resolve_topic_name("camera_info", false);

  rclcpp::QoS image_qos(1), info_qos(1);
  constexpr int throttle_period_ms = 1000;
  if (
    !find_qos(this, image_topic, image_qos, throttle_period_ms) ||
    (do_rectify && !find_qos(this, info_topic, info_qos, throttle_period_ms))) {
    RCLCPP_ERROR_THROTTLE(
      this->get_logger(), *this->get_clock(), throttle_period_ms,
      "Failed to find image QoS settings");
    return;
  }

  image_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
    image_topic, image_qos,
    [this](const sensor_msgs::msg::Image::ConstSharedPtr msg) { this->on_image(msg); });

  compressed_publisher_ =
    this->create_publisher<sensor_msgs::msg::CompressedImage>("image_raw/compressed", image_qos);

  compression_worker_.emplace(max_task_length);

  if (do_rectify) {
    info_subscription_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      info_topic, info_qos, [this](const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
        this->on_camera_info(msg);
      });

    rectified_raw_publisher_ =
      this->create_publisher<sensor_msgs::msg::Image>("image_rect", image_qos);
    rectified_info_publisher_ =
      this->create_publisher<sensor_msgs::msg::CameraInfo>("image_rect/camera_info", info_qos);
    rectified_compressed_publisher_ =
      this->create_publisher<sensor_msgs::msg::CompressedImage>("image_rect/compressed", image_qos);

    rectification_worker_.emplace(max_task_length);
  }

  // once all queries received, stop the timer callback
  qos_request_timer_->cancel();
}

void ImgProcNode::on_image(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  const auto image = std::make_shared<const common::Image>(from_ros_raw(*msg));

  // raw image compression
  if (compression_worker_) {
    // NOTE: capture `msg` by value to extend the lifetime of the shared pointer at least until the
    // task is completed
    compression_worker_->add_task([this, image, msg]() { raw_compressor_->process(*image); });
  }

  // raw image rectification and compression
  if (rectification_worker_) {
    // NOTE: capture `msg` by value to extend the lifetime of the shared pointer at least until the
    // task is completed
    rectification_worker_->add_task([this, image, msg]() {
      const auto rectified = raw_rectifier_->process(*image);
      if (rectified) {
        rectified_compressor_->process(rectified.value());
      }
    });
  }
}

void ImgProcNode::on_camera_info(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg)
{
  auto camera_info = from_ros_info(*msg);
  raw_rectifier_->set_camera_info(camera_info);
  info_subscription_.reset();
}

void ImgProcNode::publish_compressed(const common::Image & image)
{
  auto compressed = to_ros_compressed(image);
  compressed_publisher_->publish(std::move(compressed));
}

void ImgProcNode::publish_rectified_raw(const common::Image & image)
{
  auto raw = to_ros_raw(image);
  auto info = to_ros_info(raw_rectifier_->camera_info().value());
  rectified_raw_publisher_->publish(std::move(raw));
  rectified_info_publisher_->publish(std::move(info));
}

void ImgProcNode::publish_rectified_compressed(const common::Image & image)
{
  auto compressed = to_ros_compressed(image);
  rectified_compressed_publisher_->publish(std::move(compressed));
}
}  // namespace accelerated_image_processor::ros

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(accelerated_image_processor::ros::ImgProcNode)
