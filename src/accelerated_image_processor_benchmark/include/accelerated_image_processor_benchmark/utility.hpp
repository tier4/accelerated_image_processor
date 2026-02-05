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
#include <accelerated_image_processor_common/processor.hpp>

#include <yaml-cpp/node/node.h>

#include <ostream>
#include <string>
#include <vector>

namespace accelerated_image_processor::benchmark
{
/**
 * @brief Load configuration from a ROS parameter YAML file.
 * @param filepath Path to the YAML file.
 * @return YAML::Node representing the loaded configuration.
 */
YAML::Node load_config(const std::string & filepath);

/**
 * @brief Load images from a ROS bag file.
 * @param bag_path Path to the ROS bag file.
 * @param storage_id ID of the storage to use.
 * @param topic Topic to subscribe to.
 * @param num_iterations Number of iterations to load images.
 * @return Vector of loaded images.
 */
std::vector<common::Image> load_images(
  const std::string & bag_path, const std::string & storage_id, const std::string & topic,
  const int num_iterations);

/**
 * @brief Load images from a random generator.
 * @param height Height of the images.
 * @param width Width of the images.
 * @param seed Seed for the random generator.
 * @param num_iterations Number of iterations to load images.
 * @return Vector of loaded images.
 */
std::vector<common::Image> load_images(
  const int height, const int width, const int seed, const int num_iterations);

/**
 * @brief Fetch parameters from a configuration node and apply them to a processor.
 * @param config YAML::Node containing the configuration parameters.
 * @param processor Pointer to the processor to apply the parameters to.
 */
void fetch_parameters(const YAML::Node & config, common::BaseProcessor * processor);

/**
 * @brief Print information about a processor.
 * @param processor Pointer to the processor to print information about.
 */
void print_processor(const common::BaseProcessor * processor);
}  // namespace accelerated_image_processor::benchmark
