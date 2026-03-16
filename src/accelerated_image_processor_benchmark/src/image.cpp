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

#include "accelerated_image_processor_benchmark/image.hpp"

#include "accelerated_image_processor_benchmark/rosbag.hpp"

#include <accelerated_image_processor_ros/conversion.hpp>

#include <sensor_msgs/msg/image.hpp>

#include <algorithm>
#include <random>
#include <string>
#include <vector>

namespace accelerated_image_processor::benchmark
{
std::vector<common::Image> load_images(
  const std::string & bag_path, const std::string & storage_id, const std::string & topic,
  const size_t num_iteration)
{
  RosBagReader reader(bag_path, storage_id);
  const auto image_msgs = reader.read_messages<sensor_msgs::msg::Image>(topic, num_iteration);
  std::vector<common::Image> images;
  for (const auto & msg : image_msgs) {
    images.push_back(ros::from_ros_raw(msg));
  }
  return images;
}

std::vector<common::Image> load_images(
  const int height, const int width, const size_t seed, const size_t num_iteration)
{
  std::vector<common::Image> images;
  for (size_t i = 0; i < std::max(size_t(1), num_iteration); ++i) {
    images.push_back(make_synthetic_image(height, width, common::ImageEncoding::RGB, seed, i));
  }
  return images;
}

common::Image make_synthetic_image(
  int height, int width, common::ImageEncoding encoding, size_t seed, size_t index)
{
  common::Image image;
  image.frame_id = "benchmark";
  image.timestamp = 0;
  image.height = static_cast<uint32_t>(height);
  image.width = static_cast<uint32_t>(width);
  image.step = static_cast<uint32_t>(width * 3);
  image.encoding = encoding;
  image.format = common::ImageFormat::RAW;
  image.data.resize(static_cast<size_t>(image.step) * static_cast<size_t>(image.height));

  std::mt19937_64 rng(seed ^ (index + 0x9E3779B97F4A7C15ULL));
  std::uniform_int_distribution<int> noise(-12, 12);

  for (int y = 0; y < height; ++y) {
    uint8_t * row = image.data.data() + static_cast<size_t>(y * image.step);
    for (int x = 0; x < width; ++x) {
      const int base0 = (x * 255) / std::max(1, width - 1);
      const int base1 = (y * 255) / std::max(1, height - 1);
      const int base2 = ((x + y) * 255) / std::max(1, (width + height - 2));
      const int n0 = noise(rng);
      const int n1 = noise(rng);
      const int n2 = noise(rng);

      auto clamp_u8 = [](int v) -> uint8_t {
        if (v < 0) return 0;
        if (v > 255) return 255;
        return static_cast<uint8_t>(v);
      };

      row[x * 3 + 0] = clamp_u8(base0 + n0);
      row[x * 3 + 1] = clamp_u8(base1 + n1);
      row[x * 3 + 2] = clamp_u8(base2 + n2);
    }
  }

  return image;
}
}  // namespace accelerated_image_processor::benchmark
