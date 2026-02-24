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
#include "accelerated_image_processor_benchmark/image.hpp"
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
  // for warmup and iterations
  command->add_argument("--warmup").default_value(size_t{10}).nargs(1).help("Number of warmups");
  command->add_argument("--iteration")
    .default_value(size_t{100})
    .nargs(1)
    .help("Number of executions.");
  // for rosbag images
  command->add_argument("--bag").help("Directory path to the input rosbags");
  command->add_argument("--storage-id")
    .choices("sqlite3", "mcap")
    .default_value("sqlite3")
    .nargs(1)
    .help("Storage ID [required if --bag]");
  command->add_argument("--topic").help("Image topic name [required if --bag]");
  // for synthetic images
  command->add_argument("--height")
    .default_value(int{1080})
    .nargs(1)
    .help("Image height [required if --bag is NULL]");
  command->add_argument("--width").default_value(int{1920}).nargs(1).help(
    "Image width [required if --bag is NULL]");
  command->add_argument("--seed")
    .default_value(uint64_t{1})
    .nargs(1)
    .help("Random seed [required if --bag is NULL]");

  return command;
}

void run_compression(const argparse::ArgumentParser & command)
{
  // Read arguments
  const auto config_path = command.get<std::string>("config");
  const auto num_warmup = command.get<size_t>("--warmup");
  const auto num_iteration = command.get<size_t>("--iteration");

  // Load config from ROS parameter YAML file
  const auto config = load_config(config_path)["compressor"];

  // Build compressor
  auto compressor = compression::create_compressor(config["type"].as<std::string>());
  if (!compressor) {
    throw std::runtime_error("Failed to create compressor");
  }

  // Observe output sizes via postprocess callback
  Benchmarker benchmarker(config, std::move(compressor));

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

    images = load_images(bag_dir, storage_id, topic, num_iteration);
  } else {
    const auto height = command.get<int>("--height");
    const auto width = command.get<int>("--width");
    const auto seed = command.get<uint64_t>("--seed");
    std::cout << "Loading synthetic images:\n";
    std::cout << "  (Height, Width): (" << height << ", " << width << ")\n";
    images = load_images(height, width, seed, num_iteration);
  }

  // Run benchmark
  benchmarker.run(images, num_warmup, num_iteration);
}
}  // namespace accelerated_image_processor::benchmark
