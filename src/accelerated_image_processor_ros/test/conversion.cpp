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

#include <sensor_msgs/distortion_models.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace accelerated_image_processor::ros
{
TEST(TestConversionFromRosTime, Zero)
{
  const auto t = static_cast<builtin_interfaces::msg::Time>(rclcpp::Time(0, 0));

  const auto result = from_ros_time(t);

  EXPECT_EQ(result, 0LL);
}

TEST(TestConversionFromRosTime, NonZero)
{
  const auto t = static_cast<builtin_interfaces::msg::Time>(rclcpp::Time(1, 2));

  const auto result = from_ros_time(t);

  EXPECT_EQ(result, 1000000000LL + 2LL);
}

TEST(TestConversionFromRosTime, BidirectionalConversionIsConsistent)
{
  const auto ros_time = static_cast<builtin_interfaces::msg::Time>(rclcpp::Time(1234567890));
  const auto nanoseconds = from_ros_time(ros_time);

  const auto result = to_ros_time(nanoseconds);

  EXPECT_EQ(result, ros_time);
}

TEST(TestConversionFromRosEncoding, RGB8toRGB)
{
  const auto encoding = sensor_msgs::image_encodings::RGB8;

  const auto result = from_ros_encoding(encoding);

  EXPECT_EQ(result, common::ImageEncoding::RGB);
}

TEST(TestConversionFromRosEncoding, BGR8toBGR)
{
  const auto encoding = sensor_msgs::image_encodings::BGR8;

  const auto result = from_ros_encoding(encoding);

  EXPECT_EQ(result, common::ImageEncoding::BGR);
}

TEST(TestConversionFromRosEncoding, UnsupportedThrowsRuntimeError)
{
  const auto encoding = sensor_msgs::image_encodings::MONO8;
  EXPECT_THROW(from_ros_encoding(encoding), std::runtime_error);
}

TEST(TestConversionFromRosImage, CopyFields)
{
  const auto msg =
    sensor_msgs::build<sensor_msgs::msg::Image>()
      .header(
        std_msgs::build<std_msgs::msg::Header>().stamp(rclcpp::Time(123, 456)).frame_id("camera"))
      .height(480)
      .width(640)
      .encoding(sensor_msgs::image_encodings::RGB8)
      .is_bigendian(false)
      .step(1234)
      .data({1, 2, 3, 4, 5});

  const auto result = from_ros_raw(msg);

  EXPECT_EQ(result.frame_id, msg.header.frame_id);
  EXPECT_EQ(result.timestamp, 123000000000ULL + 456ULL);
  EXPECT_EQ(result.height, msg.height);
  EXPECT_EQ(result.width, msg.width);
  EXPECT_EQ(result.step, msg.step);
  EXPECT_EQ(result.encoding, common::ImageEncoding::RGB);
  EXPECT_EQ(result.format, common::ImageFormat::RAW);
  EXPECT_EQ(result.data, msg.data);
}

TEST(TestConversionFromRosImage, UnsupportedEncodingThrowsRuntimeError)
{
  const auto msg =
    sensor_msgs::build<sensor_msgs::msg::Image>()
      .header(std_msgs::build<std_msgs::msg::Header>().stamp(rclcpp::Time(0, 0)).frame_id("camera"))
      .height(1)
      .width(1)
      .encoding(sensor_msgs::image_encodings::MONO8)  // not supported by from_ros_encoding()
      .is_bigendian(false)
      .step(1)
      .data({0});

  EXPECT_THROW(from_ros_raw(msg), std::runtime_error);
}

TEST(TestConversionFromRosCameraInfo, CopyFields)
{
  const uint32_t height = 720;
  const uint32_t width = 1280;

  const auto msg = sensor_msgs::build<sensor_msgs::msg::CameraInfo>()
                     .header(
                       std_msgs::build<std_msgs::msg::Header>()
                         .stamp(static_cast<builtin_interfaces::msg::Time>(rclcpp::Time(10, 20)))
                         .frame_id("camera"))
                     .height(height)
                     .width(width)
                     .distortion_model(sensor_msgs::distortion_models::PLUMB_BOB)
                     .d({0.0, 0.0, 0.0, 0.0, 0.0})  // (k1, k2, p1, p2, k3)
                     .k(
                       {1.0, 0.0, width / 2.0,   // (fx, 0, cx)
                        0.0, 1.0, height / 2.0,  // (0, fy, cy)
                        0.0, 0.0, 1.0})          // (0, 0, 1.0)
                     .r(
                       {1.0, 0.0, 0.0,   // (r11, r12, r13)
                        0.0, 1.0, 0.0,   // (r21, r22, r23)
                        0.0, 0.0, 1.0})  // (r31, r32, r33)
                     .p(
                       {1.0, 0.0, width / 2.0, 0.0,   // (fx', 0, cx', tx)
                        0.0, 1.0, height / 2.0, 0.0,  // (0, fy', cy', ty)
                        0.0, 0.0, 1.0, 0.0})          // (0, 0, 1.0, tz)
                     .binning_x(1)
                     .binning_y(1)
                     .roi(
                       sensor_msgs::build<sensor_msgs::msg::RegionOfInterest>()
                         .x_offset(0)
                         .y_offset(0)
                         .height(height)
                         .width(width)
                         .do_rectify(false));

  const auto result = from_ros_info(msg);

  EXPECT_EQ(result.frame_id, msg.header.frame_id);
  EXPECT_EQ(result.timestamp, 10000000000LL + 20LL);
  EXPECT_EQ(result.height, msg.height);
  EXPECT_EQ(result.width, msg.width);

  EXPECT_EQ(result.d.size(), msg.d.size());
  EXPECT_EQ(result.d, msg.d);

  EXPECT_EQ(result.k.size(), msg.k.size());
  EXPECT_EQ(result.k, msg.k);

  EXPECT_EQ(result.r.size(), msg.r.size());
  EXPECT_EQ(result.r, msg.r);

  EXPECT_EQ(result.p.size(), msg.p.size());
  EXPECT_EQ(result.p, msg.p);

  EXPECT_EQ(result.binning_x, msg.binning_x);
  EXPECT_EQ(result.binning_y, msg.binning_y);

  EXPECT_EQ(result.roi.x_offset, msg.roi.x_offset);
  EXPECT_EQ(result.roi.y_offset, msg.roi.y_offset);
  EXPECT_EQ(result.roi.height, msg.roi.height);
  EXPECT_EQ(result.roi.width, msg.roi.width);
  EXPECT_EQ(result.roi.do_rectify, msg.roi.do_rectify);
}

TEST(TestConversionFromRosDistortionModel, PlumbBob)
{
  const auto distortion_model = sensor_msgs::distortion_models::PLUMB_BOB;

  const auto result = from_ros_distortion_model(distortion_model);

  EXPECT_EQ(result, common::DistortionModel::PLUMB_BOB);
}

TEST(TestConversionFromRosDistortionModel, Equidistant)
{
  const auto distortion_model = sensor_msgs::distortion_models::EQUIDISTANT;

  const auto result = from_ros_distortion_model(distortion_model);

  EXPECT_EQ(result, common::DistortionModel::EQUIDISTANT);
}

TEST(TestConversionFromRosDistortionModel, RationalPolynomial)
{
  const auto distortion_model = sensor_msgs::distortion_models::RATIONAL_POLYNOMIAL;

  const auto result = from_ros_distortion_model(distortion_model);

  EXPECT_EQ(result, common::DistortionModel::RATIONAL_POLYNOMIAL);
}

TEST(TestConversionFromRosDistortionModel, UnsupportedThrowsRuntimeError)
{
  const auto distortion_model = "foo";

  EXPECT_THROW(from_ros_distortion_model(distortion_model), std::runtime_error);
}

TEST(TestConversionFromRosRoi, CopyFields)
{
  const auto msg = sensor_msgs::build<sensor_msgs::msg::RegionOfInterest>()
                     .x_offset(0)
                     .y_offset(0)
                     .height(480)
                     .width(640)
                     .do_rectify(false);

  const auto result = from_ros_roi(msg);

  EXPECT_EQ(result.x_offset, msg.x_offset);
  EXPECT_EQ(result.y_offset, msg.y_offset);
  EXPECT_EQ(result.height, msg.height);
  EXPECT_EQ(result.width, msg.width);
  EXPECT_EQ(result.do_rectify, msg.do_rectify);
}

TEST(TestConversionToRosTime, Zero)
{
  const auto nanoseconds = 0LL;

  const auto result = to_ros_time(nanoseconds);

  EXPECT_EQ(result.sec, 0);
  EXPECT_EQ(result.nanosec, 0u);
}

TEST(TestConversionToRosTime, NonZero)
{
  const auto nanoseconds = 1'000'000'000LL + 2LL;

  const auto result = to_ros_time(nanoseconds);

  EXPECT_EQ(result.sec, 1);
  EXPECT_EQ(result.nanosec, 2u);
}

TEST(TestConversionToRosTime, BidirectionalConversionIsConsistent)
{
  const auto nanoseconds = 123456789LL;
  const auto ros_time = to_ros_time(nanoseconds);

  const auto result = from_ros_time(ros_time);

  EXPECT_EQ(result, nanoseconds);
}

TEST(TestConversionToRosRaw, CopyFields)
{
  common::Image image;
  image.frame_id = "camera";
  image.timestamp = 123000000000LL + 456LL;
  image.height = 480;
  image.width = 640;
  image.step = 2560;
  image.encoding = common::ImageEncoding::BGR;
  image.format = common::ImageFormat::RAW;
  image.data = {9, 8, 7, 6};

  const auto result = to_ros_raw(image);

  EXPECT_EQ(result.header.frame_id, image.frame_id);
  EXPECT_EQ(result.header.stamp.sec, 123);
  EXPECT_EQ(result.header.stamp.nanosec, 456u);
  EXPECT_EQ(result.height, image.height);
  EXPECT_EQ(result.width, image.width);
  EXPECT_EQ(result.encoding, sensor_msgs::image_encodings::BGR8);
  EXPECT_EQ(result.is_bigendian, false);
  EXPECT_EQ(result.step, image.step);
  EXPECT_EQ(result.data, image.data);
}

TEST(TestConversionToRosEncoding, RGBtoRGB8)
{
  const auto encoding = common::ImageEncoding::RGB;

  const auto result = to_ros_encoding(encoding);

  EXPECT_EQ(result, sensor_msgs::image_encodings::RGB8);
}

TEST(TestConversionToRosEncoding, BGRtoBGR8)
{
  const auto encoding = common::ImageEncoding::BGR;

  const auto result = to_ros_encoding(encoding);

  EXPECT_EQ(result, sensor_msgs::image_encodings::BGR8);
}

TEST(TestConversionToRosCompressed, CopyFields)
{
  common::Image image;
  image.frame_id = "camera";
  image.timestamp = 42LL;
  image.height = 1;
  image.width = 2;
  image.step = 6;
  image.encoding = common::ImageEncoding::RGB;
  image.format = common::ImageFormat::JPEG;
  image.data = {1, 2, 3, 4, 5, 6};

  const auto result = to_ros_compressed(image);

  EXPECT_EQ(result.header.frame_id, "camera");
  EXPECT_EQ(result.header.stamp.sec, 0);
  EXPECT_EQ(result.header.stamp.nanosec, 42u);
  EXPECT_EQ(result.format, "jpeg");
  EXPECT_EQ(result.data, image.data);
}

TEST(TestConversionToRosFormat, RawThrowsInvalidArgument)
{
  const auto format = common::ImageFormat::RAW;

  EXPECT_THROW(to_ros_format(format), std::invalid_argument);
}

TEST(TestConversionToRosFormat, Jpeg)
{
  const auto format = common::ImageFormat::JPEG;

  const auto result = to_ros_format(format);

  EXPECT_EQ(result, "jpeg");
}

TEST(TestConversionToRosFormat, Png)
{
  const auto format = common::ImageFormat::PNG;

  const auto result = to_ros_format(format);

  EXPECT_EQ(result, "png");
}

TEST(TestConversionToRosFFmpeg, CopyFieldsAndVideo)
{
  common::Image image;
  image.frame_id = "camera";
  image.timestamp = 123'000'000'000LL + 456LL;
  image.height = 480;
  image.width = 640;
  image.format = common::ImageFormat::H264;
  image.data = {9, 8, 7, 6};
  image.pts = 123456789ULL;
  image.flags = 1;
  image.is_bigendian = true;

  auto pkt = to_ros_ffmpeg(image);

  EXPECT_EQ(pkt.header.frame_id, image.frame_id);
  EXPECT_EQ(pkt.header.stamp.sec, 123);
  EXPECT_EQ(pkt.header.stamp.nanosec, 456);
  EXPECT_EQ(pkt.width, image.width);
  EXPECT_EQ(pkt.height, image.height);
  EXPECT_EQ(pkt.encoding, "h264");
  EXPECT_EQ(pkt.pts, image.pts.value());
  EXPECT_EQ(pkt.flags, image.flags.value());
  EXPECT_EQ(pkt.is_bigendian, image.is_bigendian.value());
  EXPECT_EQ(pkt.data, image.data);
}

TEST(TestConversionToRosFFmpegEncoding, ValidEncodings)
{
  struct
  {
    common::ImageFormat fmt;
    std::string expected;
  } cases[] = {
    {common::ImageFormat::RAW, "raw"},   {common::ImageFormat::JPEG, "jpeg"},
    {common::ImageFormat::PNG, "png"},   {common::ImageFormat::H264, "h264"},
    {common::ImageFormat::H265, "hevc"}, {common::ImageFormat::AV1, "av1"},
  };

  auto is_supported_encoding = [](const common::ImageFormat & fmt) {
    if (
      fmt == common::ImageFormat::H264 || fmt == common::ImageFormat::H265 ||
      fmt == common::ImageFormat::AV1) {
      return true;
    } else {
      return false;
    }
  };

  for (const auto & c : cases) {
    if (is_supported_encoding(c.fmt)) {
      EXPECT_EQ(to_ros_ffmpeg_encoding(c.fmt), c.expected);
    } else {
      EXPECT_THROW(to_ros_ffmpeg_encoding(c.fmt), std::runtime_error);
    }
  }
}

TEST(TestConversionToRosInfo, CopyFields)
{
  common::CameraInfo info;
  info.frame_id = "camera";
  info.timestamp = 10'000'000'000LL + 20LL;
  info.height = 720;
  info.width = 1280;
  info.distortion_model = common::DistortionModel::PLUMB_BOB;
  info.d = {0.0, 0.0, 0.0, 0.0, 0.0};     // (k1, k2, p1, p2, k3)
  info.k = {1.0, 0.0, info.width / 2.0,   // (fx, 0, cx)
            0.0, 1.0, info.height / 2.0,  // (0, fy, cy)
            0.0, 0.0, 1.0};               // (0, 0, 1.0)
  info.r = {1.0, 0.0, 0.0,                // (r11, r12, r13)
            0.0, 1.0, 0.0,                // (r21, r22, r23)
            0.0, 0.0, 1.0};               // (r31, r32, r33)
  info.p = {1.0, 0.0, info.width / 2.0,
            0.0,  // (fx', 0, cx', tx)
            0.0, 1.0, info.height / 2.0,
            0.0,  // (0, fy', cy', ty)
            0.0, 0.0, 1.0,
            0.0};  // (0, 0, 1.0, tz)
  info.binning_x = 1;
  info.binning_y = 1;
  info.roi = common::Roi{0, 0, info.width, info.height, false};

  const auto result = to_ros_info(info);

  EXPECT_EQ(result.header.frame_id, info.frame_id);
  EXPECT_EQ(result.header.stamp.sec, 10);
  EXPECT_EQ(result.header.stamp.nanosec, 20u);

  EXPECT_EQ(result.height, info.height);
  EXPECT_EQ(result.width, info.width);

  EXPECT_EQ(result.distortion_model, sensor_msgs::distortion_models::PLUMB_BOB);
  EXPECT_EQ(result.d, info.d);
  EXPECT_EQ(result.k, info.k);
  EXPECT_EQ(result.r, info.r);
  EXPECT_EQ(result.p, info.p);

  EXPECT_EQ(result.binning_x, info.binning_x);
  EXPECT_EQ(result.binning_y, info.binning_y);

  EXPECT_EQ(result.roi.x_offset, info.roi.x_offset);
  EXPECT_EQ(result.roi.y_offset, info.roi.y_offset);
  EXPECT_EQ(result.roi.height, info.roi.height);
  EXPECT_EQ(result.roi.width, info.roi.width);
  EXPECT_EQ(result.roi.do_rectify, info.roi.do_rectify);
}

TEST(TestConversionToRosDistortionModel, PlumbBob)
{
  const auto distortion_model = common::DistortionModel::PLUMB_BOB;

  const auto result = to_ros_distortion_model(distortion_model);

  EXPECT_EQ(result, sensor_msgs::distortion_models::PLUMB_BOB);
}

TEST(TestConversionToRosDistortionModel, Equidistant)
{
  const auto distortion_model = common::DistortionModel::EQUIDISTANT;

  const auto result = to_ros_distortion_model(distortion_model);

  EXPECT_EQ(result, sensor_msgs::distortion_models::EQUIDISTANT);
}

TEST(TestConversionToRosDistortionModel, RationalPolynomial)
{
  const auto distortion_model = common::DistortionModel::RATIONAL_POLYNOMIAL;

  const auto result = to_ros_distortion_model(distortion_model);

  EXPECT_EQ(result, sensor_msgs::distortion_models::RATIONAL_POLYNOMIAL);
}

TEST(TestConversionToRosRoi, CopyFields)
{
  const auto roi = common::Roi{0, 0, 30, 40, false};

  const auto result = to_ros_roi(roi);

  EXPECT_EQ(result.x_offset, roi.x_offset);
  EXPECT_EQ(result.y_offset, roi.y_offset);
  EXPECT_EQ(result.height, roi.height);
  EXPECT_EQ(result.width, roi.width);
  EXPECT_EQ(result.do_rectify, roi.do_rectify);
}
}  // namespace accelerated_image_processor::ros

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
