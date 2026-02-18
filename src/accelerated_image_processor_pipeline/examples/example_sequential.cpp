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

common::Image make_image(int width, int height)
{
  common::Image image;
  image.frame_id = "camera";
  image.timestamp = 123456789;
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
  return image;
}
}  // namespace

void run_sequential()
{
  pipeline::Sequential sequential;

  // Pipeline: Rectification -> [Info] -> JPEG Compression -> [Info] -> [Finish]
  sequential.append<pipeline::Rectifier>("rectifier", &print_info)
    .append<compression::Compressor>("compressor", &print_info, "jpeg")
    .register_postprocess(&print_finish);

  const auto image = make_image(1920, 1080);

  sequential.process(image);
}
}  // namespace accelerated_image_processor

int main()
{
  accelerated_image_processor::run_sequential();
}
