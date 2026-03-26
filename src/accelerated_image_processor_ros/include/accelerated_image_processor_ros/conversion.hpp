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

#include <accelerated_image_processor_common/datatype.hpp>
#include <rclcpp/time.hpp>

#include <ffmpeg_image_transport_msgs/msg/ffmpeg_packet.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/region_of_interest.hpp>
#include <std_msgs/msg/header.hpp>

#include <string>
#include <vector>

namespace accelerated_image_processor::ros
{
/// === From ROS messages to common data types ===

/**
 * @brief Convert builtin_interfaces::msg::Time to int64_t in nanoseconds.
 * @param stamp Time message to convert.
 * @return int64_t in nanoseconds.
 */
int64_t from_ros_time(const builtin_interfaces::msg::Time & stamp);

/// --- From sensor_msgs::msg::Image to common::Image ---

/**
 * @brief Convert sensor_msgs::msg::Image to common::Image.
 * @param msg sensor_msgs::msg::Image message
 * @return common::Image
 */
common::Image from_ros_raw(const sensor_msgs::msg::Image & msg);

/**
 * @brief Convert sensor_msgs::msg::Image encoding to common::ImageEncoding.
 * @param encoding Encoding of sensor_msgs::msg::Image.
 * @return common::ImageEncoding
 */
common::ImageEncoding from_ros_encoding(const std::string & encoding);

///--- From sensor_msgs::msg::CameraInfo to common::CameraInfo ---

/**
 * @brief Convert sensor_msgs::msg::CameraInfo to common::CameraInfo.
 * @param msg sensor_msgs::msg::CameraInfo message
 * @return common::CameraInfo
 */
common::CameraInfo from_ros_info(const sensor_msgs::msg::CameraInfo & msg);

/**
 * @brief Convert sensor_msgs::msg::CameraInfo distortion model to common::DistortionModel.
 * @param model Distortion model of sensor_msgs::msg::CameraInfo.
 * @return common::DistortionModel
 */
common::DistortionModel from_ros_distortion_model(const std::string & model);

/**
 * @brief Convert sensor_msgs::msg::RegionOfInterest to common::Roi.
 * @param roi sensor_msgs::msg::RegionOfInterest message
 * @return common::Roi
 */
common::Roi from_ros_roi(const sensor_msgs::msg::RegionOfInterest & roi);

/// === From common data types to ROS messages ===

/**
 * @brief Return sensor_msgs::msg::Header.
 * @param timestamp Timestamp in nanoseconds.
 * @param frame_id Frame ID.
 * @return sensor_msgs::msg::Header
 */
std_msgs::msg::Header to_ros_header(int64_t timestamp, const std::string & frame_id);

/**
 * @brief Convert int64_t in nanoseconds to builtin_interfaces::msg::Time.
 * @param timestamp Timestamp in nanoseconds.
 * @return builtin_interfaces::msg::Time
 */
builtin_interfaces::msg::Time to_ros_time(int64_t timestamp);

/// --- From common::Image to sensor_msgs::msg::Image ---

/**
 * @brief Convert common::Image to sensor_msgs::msg::Image.
 * @param image common::Image message
 * @return sensor_msgs::msg::Image
 */
sensor_msgs::msg::Image to_ros_raw(const common::Image & image);

/**
 * @brief Convert common::ImageEncoding to sensor_msgs::msg::Image encoding.
 * @param encoding Encoding of common::Image.
 * @return std::string
 */
std::string to_ros_encoding(common::ImageEncoding encoding);

/// --- From common::Image to sensor_msgs::msg::CompressedImage ---

/**
 * @brief Convert common::Image to sensor_msgs::msg::CompressedImage.
 * @param image common::Image message
 * @return sensor_msgs::msg::CompressedImage
 */
sensor_msgs::msg::CompressedImage to_ros_compressed(const common::Image & image);

/**
 * @brief Convert common::ImageFormat to sensor_msgs::msg::Image format.
 * @param format Format of common::Image.
 * @return std::string
 */
std::string to_ros_format(common::ImageFormat format);

/// --- From common::CameraInfo to sensor_msgs::msg::CameraInfo ---

/**
 * @brief Convert common::CameraInfo to sensor_msgs::msg::CameraInfo.
 * @param info common::CameraInfo message
 * @return sensor_msgs::msg::CameraInfo
 */
sensor_msgs::msg::CameraInfo to_ros_info(const common::CameraInfo & info);

/**
 * @brief Convert common::DistortionModel to sensor_msgs::msg::CameraInfo distortion_model.
 * @param model DistortionModel of common::CameraInfo.
 * @return std::string
 */
std::string to_ros_distortion_model(common::DistortionModel model);

/**
 * @brief Convert common::Roi to sensor_msgs::msg::RegionOfInterest.
 * @param roi common::Roi message
 * @return sensor_msgs::msg::RegionOfInterest
 */
sensor_msgs::msg::RegionOfInterest to_ros_roi(const common::Roi & roi);

/// --- From common::Image to ffmpeg_image_transport_msgs::msg::FFMPEGPacket ---

/**
 * @brief Convert common::Image to ffmpeg_image_transport_msgs::msg::FFMPEGPacket.
 * @param image common::Image message
 * @return ffmpeg_image_transport_msgs::msg::FFMPEGPacket
 */
ffmpeg_image_transport_msgs::msg::FFMPEGPacket to_ros_ffmpeg(const common::Image & image);

/**
 * @brief Convert common::ImageFormat to ffmpeg_image_transport_msgs::msg::FFMPEGPacket encoding.
 * @param format Format of common::Image.
 * @return std::string
 */
std::string to_ros_ffmpeg_encoding(common::ImageFormat format);

/// --- From ffmpeg_image_transport_msgs::msg::FFMPEGPacket to common::Image ---
/**
 * @brief split string by ',' or ';' and return separated elements
 * @param input_string std::string to be split
 * @return std::vector<std::string>
 */
std::vector<std::string> split_string_by_comma_and_semicolon(const std::string & input_string);

/**
 * @brief Convert std::string to common::ImageFormat
 * @param encoding_str std::string to be converted
 * @return common::ImageFormat
 */
common::ImageFormat from_ros_ffmpeg_encoding(const std::string & encoding_str);

/**
 * @brief Convert ffmpeg_image_transport_msgs::msg::FFMPEGPacket to common::Image
 * @param msg ffmpeg_image_transport_msgs::msg::FFMPEGPacket to be converted
 * @return common::Image
 */
common::Image from_ros_ffmpeg(const ffmpeg_image_transport_msgs::msg::FFMPEGPacket & msg);

}  // namespace accelerated_image_processor::ros
