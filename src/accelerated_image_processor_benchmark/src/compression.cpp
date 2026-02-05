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

#include <exception>
#include <iostream>
#include <string>
#include <utility>
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

  // Read arguments
  const auto config_path = program.get<std::string>("config");
  const auto num_warmups = program.get<int>("--warmup");
  const auto num_iterations = program.get<int>("--iterations");

  // Load config from ROS parameter YAML file
  const auto config = benchmark::load_config(config_path)["compressor"];

  // Build compressor
  auto compressor = compression::create_compressor(config["type"].as<std::string>());
  if (!compressor) {
    throw std::runtime_error("Failed to create compressor");
  }

  // Observe output sizes via postprocess callback
  benchmark::Benchmarker benchmarker(config, std::move(compressor));

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

    images = benchmark::load_images(bag_dir, storage_id, topic);
  } else {
    const auto height = program.get<int>("--height");
    const auto width = program.get<int>("--width");
    const auto seed = program.get<int>("--seed");
    std::cout << "Loading synthetic images:\n";
    std::cout << "  (Height, Width): (" << height << ", " << width << ")\n";
    images = benchmark::load_images(height, width, num_iterations, seed);
  }

  // Run benchmark
  benchmarker.run(images, num_warmups, num_iterations);

  return 0;
}
