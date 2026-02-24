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

#include "accelerated_image_processor_benchmark/utility.hpp"

#include <accelerated_image_processor_common/processor.hpp>
#include <rmw/impl/cpp/demangle.hpp>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <variant>

#ifdef __GNUG__
#include <cxxabi.h>

#include <cstdlib>
#endif

namespace accelerated_image_processor::benchmark
{
namespace
{
#ifdef __GNUG__
/**
 * @brief Demangle a C++ type name for GCC/Clang compiler.
 *
 * @param name The mangled type name.
 * @return std::string The demangled type name.
 */
std::string demangle(const char * name)
{
  int status = -1;
  std::unique_ptr<char, void (*)(void *)> res{
    abi::__cxa_demangle(name, NULL, NULL, &status), std::free};
  return (status == 0) ? res.get() : name;
}
#else
/**
 * @brief Demangle a C++ type name for non-GCC/Clang compiler.
 *
 * @param name The mangled type name.
 * @return std::string The demangled type name.
 */
std::string demangle(const char * name)
{
  return name;
}
#endif
}  // namespace

YAML::Node load_config(const std::string & filepath)
{
  YAML::Node config = YAML::LoadFile(filepath);
  if (!config) {
    throw std::runtime_error("Failed to load config file");
  }
  return config["/**"]["ros__parameters"];
}

void fetch_parameters(const YAML::Node & config, common::BaseProcessor * processor)
{
  // Fetch parameters from config
  for (auto & [name, value] : processor->parameters()) {
    const auto & param_name = name;
    std::visit(
      [&](auto & v) {
        using T = std::decay_t<decltype(v)>;
        if (config[param_name]) {
          v = config[param_name].as<T>();
        }
      },
      value);
  }
}

void print_processor(const common::BaseProcessor * processor)
{
  std::cout << "------------------ Processor Information ------------------\n";

  std::cout << "ClassName: " << demangle(typeid(*processor).name()) << "\n";

  std::cout << "Parameters:\n";
  for (const auto & [name, value] : processor->parameters()) {
    const auto param_name = name;
    std::visit([&](auto & v) { std::cout << "  " << param_name << ": " << v << "\n"; }, value);
  }

  std::cout << "-----------------------------------------------------------\n";
}
}  // namespace accelerated_image_processor::benchmark
