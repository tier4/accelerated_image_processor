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

#include "accelerated_image_processor_ros/conversion.hpp"

#include <accelerated_image_processor_common/datatype.hpp>
#include <rclcpp/time.hpp>

#include <sensor_msgs/distortion_models.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <stdexcept>
#include <string>

namespace accelerated_image_processor::ros
{
/// === From ROS messages to common data types ===

int64_t from_ros_time(const builtin_interfaces::msg::Time & stamp)
{
  return rclcpp::Time(stamp.sec, stamp.nanosec).nanoseconds();
}

/// --- From sensor_msgs::msg::Image to common::Image ---

common::Image from_ros_raw(const sensor_msgs::msg::Image & msg)
{
  common::Image output;
  output.frame_id = msg.header.frame_id;
  output.timestamp = from_ros_time(msg.header.stamp);
  output.height = msg.height;
  output.width = msg.width;
  output.step = msg.step;
  output.encoding = from_ros_encoding(msg.encoding);
  output.format = common::ImageFormat::RAW;
  output.data = msg.data;
  return output;
}

common::ImageEncoding from_ros_encoding(const std::string & encoding)
{
  if (encoding == sensor_msgs::image_encodings::RGB8) {
    return common::ImageEncoding::RGB;
  } else if (encoding == sensor_msgs::image_encodings::BGR8) {
    return common::ImageEncoding::BGR;
  } else {
    throw std::runtime_error("Unsupported encoding: " + encoding);
  }
}

///--- From sensor_msgs::msg::CameraInfo to common::CameraInfo ---

common::CameraInfo from_ros_info(const sensor_msgs::msg::CameraInfo & msg)
{
  common::CameraInfo output;
  output.frame_id = msg.header.frame_id;
  output.timestamp = from_ros_time(msg.header.stamp);
  output.height = msg.height;
  output.width = msg.width;
  output.distortion_model = from_ros_distortion_model(msg.distortion_model);
  output.d = msg.d;
  output.k = msg.k;
  output.r = msg.r;
  output.p = msg.p;
  output.binning_x = msg.binning_x;
  output.binning_y = msg.binning_y;
  output.roi = from_ros_roi(msg.roi);
  return output;
}

common::DistortionModel from_ros_distortion_model(const std::string & model)
{
  if (model == sensor_msgs::distortion_models::PLUMB_BOB) {
    return common::DistortionModel::PLUMB_BOB;
  } else if (model == sensor_msgs::distortion_models::EQUIDISTANT) {
    return common::DistortionModel::EQUIDISTANT;
  } else if (model == sensor_msgs::distortion_models::RATIONAL_POLYNOMIAL) {
    return common::DistortionModel::RATIONAL_POLYNOMIAL;
  } else {
    throw std::runtime_error("Unsupported distortion model: " + model);
  }
}

common::Roi from_ros_roi(const sensor_msgs::msg::RegionOfInterest & roi)
{
  return common::Roi{roi.x_offset, roi.y_offset, roi.width, roi.height, roi.do_rectify};
}

/// === From common data types to ROS messages ===

std_msgs::msg::Header to_ros_header(int64_t timestamp, const std::string & frame_id)
{
  return std_msgs::build<std_msgs::msg::Header>().stamp(to_ros_time(timestamp)).frame_id(frame_id);
}

builtin_interfaces::msg::Time to_ros_time(int64_t timestamp)
{
  rclcpp::Time rclcpp_time(timestamp);
  return static_cast<builtin_interfaces::msg::Time>(rclcpp_time);
}

/// --- From common::Image to sensor_msgs::msg::Image ---

sensor_msgs::msg::Image to_ros_raw(const common::Image & image)
{
  return sensor_msgs::build<sensor_msgs::msg::Image>()
    .header(to_ros_header(image.timestamp, image.frame_id))
    .height(image.height)
    .width(image.width)
    .encoding(to_ros_encoding(image.encoding))
    .is_bigendian(false)
    .step(image.step)
    .data(image.data);
}

std::string to_ros_encoding(common::ImageEncoding encoding)
{
  switch (encoding) {
    case common::ImageEncoding::RGB:
      return sensor_msgs::image_encodings::RGB8;
    case common::ImageEncoding::BGR:
      return sensor_msgs::image_encodings::BGR8;
    default:
      throw std::runtime_error("Unsupported format: " + std::to_string(static_cast<int>(encoding)));
  }
}

/// --- From common::Image to sensor_msgs::msg::CompressedImage ---

sensor_msgs::msg::CompressedImage to_ros_compressed(const common::Image & image)
{
  return sensor_msgs::build<sensor_msgs::msg::CompressedImage>()
    .header(to_ros_header(image.timestamp, image.frame_id))
    .format(to_ros_format(image.format))
    .data(image.data);
}

std::string to_ros_format(common::ImageFormat format)
{
  switch (format) {
    case common::ImageFormat::RAW:
      throw std::invalid_argument("Raw image is not supported");
    case common::ImageFormat::JPEG:
      return "jpeg";
    case common::ImageFormat::PNG:
      return "png";
    default:
      throw std::runtime_error("Unsupported format: " + std::to_string(static_cast<int>(format)));
  }
}

/// --- From common::CameraInfo to sensor_msgs::msg::CameraInfo ---

sensor_msgs::msg::CameraInfo to_ros_info(const common::CameraInfo & info)
{
  return sensor_msgs::build<sensor_msgs::msg::CameraInfo>()
    .header(to_ros_header(info.timestamp, info.frame_id))
    .height(info.height)
    .width(info.width)
    .distortion_model(to_ros_distortion_model(info.distortion_model))
    .d(info.d)
    .k(info.k)
    .r(info.r)
    .p(info.p)
    .binning_x(info.binning_x)
    .binning_y(info.binning_y)
    .roi(to_ros_roi(info.roi));
}

std::string to_ros_distortion_model(common::DistortionModel model)
{
  switch (model) {
    case common::DistortionModel::PLUMB_BOB:
      return sensor_msgs::distortion_models::PLUMB_BOB;
    case common::DistortionModel::EQUIDISTANT:
      return sensor_msgs::distortion_models::EQUIDISTANT;
    case common::DistortionModel::RATIONAL_POLYNOMIAL:
      return sensor_msgs::distortion_models::RATIONAL_POLYNOMIAL;
    default:
      throw std::runtime_error(
        "Unsupported distortion model: " + std::to_string(static_cast<int>(model)));
  }
}

sensor_msgs::msg::RegionOfInterest to_ros_roi(const common::Roi & roi)
{
  return sensor_msgs::build<sensor_msgs::msg::RegionOfInterest>()
    .x_offset(roi.x_offset)
    .y_offset(roi.y_offset)
    .height(roi.height)
    .width(roi.width)
    .do_rectify(roi.do_rectify);
}
}  // namespace accelerated_image_processor::ros
