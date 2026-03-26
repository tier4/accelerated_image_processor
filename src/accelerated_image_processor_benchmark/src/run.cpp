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

#include <argparse/argparse.hpp>

#include <exception>
#include <iostream>

int main(int argc, char ** argv)
{
  using namespace accelerated_image_processor;  // NOLINT

  argparse::ArgumentParser program("accbench");
  program.add_description("Benchmark CLI");

  // Add subcommands
  const auto compression_command = benchmark::make_compression_command();
  program.add_subparser(*compression_command);

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error & err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  if (program.is_subcommand_used("compression")) {
    benchmark::run_compression(*compression_command);
    return 0;
  } else {
    std::cerr << "No valid subcommand exists" << std::endl;
    return 1;
  }
}
