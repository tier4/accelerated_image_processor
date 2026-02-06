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

#include "accelerated_image_processor_benchmark/compression.hpp"

#include "accelerated_image_processor_benchmark/benchmarker.hpp"
#include "accelerated_image_processor_benchmark/utility.hpp"

#include <accelerated_image_processor_compression/builder.hpp>
#include <argparse/argparse.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace accelerated_image_processor::benchmark
{
std::shared_ptr<argparse::ArgumentParser> make_compression_command()
{
  auto command = std::make_shared<argparse::ArgumentParser>("compression");
  command->add_description("Benchmark CLI for image compression");
  command->add_argument("config").required().help("Filepath to the configuration file");
  // for rosbag images
  command->add_argument("--bag").help("Directory path to the input rosbags");
  command->add_argument("--storage-id")
    .choices("sqlite3", "mcap")
    .default_value("sqlite3")
    .help("Storage ID, only required if --bag is specified");
  command->add_argument("--topic").help("Image topic name, only required if --bag is specified");
  // for synthetic images
  command->add_argument("--height")
    .default_value(1080)
    .help("Image height, only required if --bag is not specified");
  command->add_argument("--width").default_value(1920).help(
    "Image width, only required if --bag is not specified");
  command->add_argument("--seed").default_value(1).help(
    "Random seed, only required if --bag is not specified");
  // for warmup and iterations
  command->add_argument("--warmup").default_value(10).help("Number of warmup iterations");
  command->add_argument("--iterations").default_value(100);

  return command;
}

void run_compression(const argparse::ArgumentParser & command)
{
  // Read arguments
  const auto config_path = command.get<std::string>("config");
  const auto num_warmups = command.get<int>("--warmup");
  const auto num_iterations = command.get<int>("--iterations");

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
  if (command.is_used("--bag")) {
    if (!command.is_used("--topic")) {
      throw std::runtime_error("Topic is required when using bag");
    }
    const auto bag_dir = command.get<std::string>("--bag");
    const auto storage_id = command.get<std::string>("--storage-id");
    const auto topic = command.get<std::string>("--topic");

    std::cout << "Loading images from bag:\n"
              << "  Bag: " << bag_dir << "\n"
              << "  Storage ID: " << storage_id << "\n"
              << "  Topic: " << topic << "\n";

    images = benchmark::load_images(bag_dir, storage_id, topic, num_iterations);
  } else {
    const auto height = command.get<int>("--height");
    const auto width = command.get<int>("--width");
    const auto seed = command.get<int>("--seed");
    std::cout << "Loading synthetic images:\n";
    std::cout << "  (Height, Width): (" << height << ", " << width << ")\n";
    images = benchmark::load_images(height, width, seed, num_iterations);
  }

  // Run benchmark
  benchmarker.run(images, num_warmups, num_iterations);
}
}  // namespace accelerated_image_processor::benchmark
