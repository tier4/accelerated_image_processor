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

#include "accelerated_image_processor_benchmark/benchmarker.hpp"
#include "accelerated_image_processor_benchmark/image.hpp"
#include "accelerated_image_processor_benchmark/rosbag.hpp"
#include "accelerated_image_processor_benchmark/utility.hpp"

#include <accelerated_image_processor_compression/builder.hpp>
#include <accelerated_image_processor_ros/conversion.hpp>
#include <argparse/argparse.hpp>

#include <sensor_msgs/msg/image.hpp>

#include <algorithm>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

using namespace accelerated_image_processor;  // NOLINT

int main(int argc, char ** argv)
{
  argparse::ArgumentParser program("compression");
  program.add_description("Benchmark CLI for image compression");
  program.add_argument("config").required().help("Filepath to the configuration file");
  // for rosbag images
  program.add_argument("--bag").help("Directory path to the input rosbags");
  program.add_argument("--storage-id")
    .choices("sqlite3", "mcap")
    .default_value("sqlite3")
    .help("Storage ID, only required if --bag is specified");
  program.add_argument("--topic").help("Image topic name, only required if --bag is specified");
  // for synthetic images
  program.add_argument("--height")
    .default_value(1080)
    .help("Image height, only required if --bag is not specified");
  program.add_argument("--width").default_value(1920).help(
    "Image width, only required if --bag is not specified");
  program.add_argument("--seed").default_value(1).help(
    "Random seed, only required if --bag is not specified");
  // for warmup and iterations
  program.add_argument("--warmup").default_value(10).help("Number of warmup iterations");
  program.add_argument("--iterations").default_value(100);

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception & err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  // Load config from ROS parameter YAML file
  const auto config = benchmark::load_config(program.get<std::string>("config"))["compressor"];

  // Build compressor
  auto compressor = compression::create_compressor(config["type"].as<std::string>());
  if (!compressor) {
    throw std::runtime_error("Failed to create compressor");
  }

  // Fetch parameters from config
  benchmark::fetch_parameters(config, compressor.get());

  // Print processor information
  benchmark::print_processor(compressor.get());

  // Observe output sizes via postprocess callback
  benchmark::Benchmarker benchmarker;
  compressor->register_postprocess<benchmark::Benchmarker, &benchmark::Benchmarker::on_image>(
    &benchmarker);

  // Prepare input images
  std::vector<common::Image> images;
  std::cout << ">>> Preparing input images\n";
  if (program.is_used("--bag")) {
    if (!program.is_used("--topic")) {
      throw std::runtime_error("Topic is required when using bag");
    }
    const auto bag_dir = program.get<std::string>("--bag");
    const auto storage_id = program.get<std::string>("--storage-id");
    const auto topic = program.get<std::string>("--topic");

    std::cout << "Loading images from bag:\n"
              << "  Bag: " << bag_dir << "\n"
              << "  Storage ID: " << storage_id << "\n"
              << "  Topic: " << topic << "\n";

    benchmark::RosBagReader reader(bag_dir, storage_id);

    const auto image_msgs = reader.read_messages<sensor_msgs::msg::Image>(topic);
    for (const auto & msg : image_msgs) {
      images.push_back(ros::from_ros_raw(msg));
    }
  } else {
    const auto height = program.get<int>("--height");
    const auto width = program.get<int>("--width");
    const auto seed = program.get<int>("--seed");
    std::cout << "Loading synthetic images:\n";
    std::cout << "  (Height, Width): (" << height << ", " << width << ")\n";
    for (int i = 0; i < std::max(1, program.get<int>("--iterations")); ++i) {
      images.push_back(
        benchmark::make_synthetic_image(height, width, common::ImageEncoding::RGB, seed, i));
    }
  }

  // Set source image bytes
  benchmarker.set_source_bytes(images);

  auto try_processing = [&images, &compressor](size_t iter_idx) {
    const auto idx = iter_idx % images.size();
    compressor->process(images[idx]);
  };

  // Warmup
  const auto warmup = program.get<int>("--warmup");
  std::cout << ">>> Starting warmup [n = " << warmup << "]" << std::endl;
  for (int i = 0; i < warmup; ++i) {
    try_processing(static_cast<size_t>(i));
  }
  std::cout << "<<< ✨Finished warmup" << std::endl;

  // Reset benchmarker and run iterations
  const auto iterations = program.get<int>("--iterations");
  std::cout << ">>> Starting iterations [n = " << iterations << "]" << std::endl;
  benchmarker.reset_processed();
  for (int i = 0; i < iterations; ++i) {
    benchmarker.tic();
    try_processing(static_cast<size_t>(i));
  }
  std::cout << "<<< ✨Finished iterations" << std::endl;

  // Print benchmark results
  benchmarker.print();

  return 0;
}
