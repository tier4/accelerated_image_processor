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

#include <algorithm>
#include <string>
#include <unordered_map>
#include <variant>

namespace accelerated_image_processor
{
namespace common
{
/**
 * @brief Type alias for parameter keys.
 */
using ParameterKey = std::string;

/**
 * @brief Type alias for parameter values.
 */
using ParameterValue = std::variant<int, double, bool, std::string>;

/**
 * @brief Parameter map type, which is a map of parameter names to parameter values.
 */
using ParameterMap = std::unordered_map<ParameterKey, ParameterValue>;
}  // namespace common

/**
 * @brief Merge two parameter maps.
 * @param lhs The left-hand side parameter map.
 * @param rhs The right-hand side parameter map.
 * @return common::ParameterMap& The merged parameter map.
 */
inline common::ParameterMap & operator+=(
  common::ParameterMap & lhs, const common::ParameterMap & rhs)
{
  for (const auto & [key, value] : rhs) {
    lhs[key] = value;
  }
  return lhs;
}
}  // namespace accelerated_image_processor
