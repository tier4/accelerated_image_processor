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

#include "accelerated_image_processor_pipeline/rectifier.hpp"
#include "accelerated_image_processor_pipeline/sequential.hpp"

#include <accelerated_image_processor_common/datatype.hpp>
#include <accelerated_image_processor_compression/builder.hpp>

#include <iostream>
#include <string>
#include <utility>

namespace accelerated_image_processor
{
namespace
{
std::string format_to_str(common::ImageFormat format)
{
  switch (format) {
    case common::ImageFormat::RAW:
      return "RAW";
    case common::ImageFormat::JPEG:
      return "JPEG";
    default:
      return "UNKNOWN";
  }
}

void print_info(const common::Image & image)
{
  std::cout << "[INFO]:\n"
            << "  (width, height) = (" << image.width << ", " << image.height << ")\n"
            << "  format = " << format_to_str(image.format) << std::endl;
}

void print_finish(const common::Image &)
{
  std::cout << ">>> 🎉 All processing finished!!" << std::endl;
}

std::pair<common::Image, common::CameraInfo> make_image_and_info(uint32_t width, uint32_t height)
{
  const std::string frame_id = "camera";
  constexpr int64_t timestamp = 123456789;

  common::Image image;
  image.frame_id = frame_id;
  image.timestamp = timestamp;
  image.width = width;
  image.height = height;
  image.step = width * 3;
  image.encoding = common::ImageEncoding::RGB;
  image.format = common::ImageFormat::RAW;
  image.data.resize(image.step * image.height);

  for (uint32_t y = 0; y < image.height; ++y) {
    for (uint32_t x = 0; x < image.width; ++x) {
      image.data[y * image.step + x * 3 + 0] = static_cast<uint8_t>(x % 256);
      image.data[y * image.step + x * 3 + 1] = static_cast<uint8_t>(y % 256);
      image.data[y * image.step + x * 3 + 2] = static_cast<uint8_t>((x + y) % 256);
    }
  }

  common::CameraInfo camera_info;
  camera_info.frame_id = frame_id;
  camera_info.timestamp = timestamp;
  camera_info.width = width;
  camera_info.height = height;
  camera_info.distortion_model = common::DistortionModel::PLUMB_BOB;
  camera_info.d = {0.0, 0.0, 0.0, 0.0, 0.0};      // (k1, k2, p1, p2, k3)
  camera_info.k = {1.0, 0.0, width / 2.0,         // (fx, 0, cx)
                   0.0, 1.0, height / 2.0,        // (0, fy, cy)
                   0.0, 0.0, 1.0};                // (0, 0, 1.0)
  camera_info.r = {1.0, 0.0, 0.0,                 // (r11, r12, r13)
                   0.0, 1.0, 0.0,                 // (r21, r22, r23)
                   0.0, 0.0, 1.0};                // (r31, r32, r33)
  camera_info.p = {1.0, 0.0, width / 2.0,  0.0,   // (fx', 0, cx', tx)
                   0.0, 1.0, height / 2.0, 0.0,   // (0, fy', cy', ty)
                   0.0, 0.0, 1.0,          0.0};  // (0, 0, 1.0, tz)
  camera_info.binning_x = 1;
  camera_info.binning_y = 1;
  camera_info.roi = {0, 0, width, height, false};
  return std::make_pair(image, camera_info);
}
}  // namespace

void run_sequential()
{
  const auto [image, camera_info] = make_image_and_info(1920, 1080);

  pipeline::Sequential sequential;

  // Pipeline: Rectification -> [Info] -> JPEG Compression -> [Info] -> [Finish]
  sequential.append<pipeline::Rectifier>("rectifier", &print_info)
    .append<compression::Compressor>("compressor", &print_info, "jpeg")
    .register_postprocess(&print_finish);

  // Set camera info for rectification
  sequential.set_camera_info(camera_info);

  sequential.process(image);
}
}  // namespace accelerated_image_processor

int main()
{
  accelerated_image_processor::run_sequential();
}
