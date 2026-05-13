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

#include <argparse/argparse.hpp>

#include <memory>

namespace accelerated_image_processor::benchmark
{
/**
 * @brief Create a compression command.
 * @return std::shared_ptr<argparse::ArgumentParser> Shared pointer to the created command.
 */
std::shared_ptr<argparse::ArgumentParser> make_compression_command();

/**
 * @brief Run the compression benchmark.
 * @param command The argument parser.
 */
void run_compression(const argparse::ArgumentParser & command);
}  // namespace accelerated_image_processor::benchmark
