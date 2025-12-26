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

#include "accelerated_image_processor_pipeline/rectifier.hpp"

#include <accelerated_image_processor_common/datatype.hpp>

#include <memory>
#include <type_traits>

namespace accelerated_image_processor::pipeline
{
/**
 * @brief Create a rectifier object.
 *
 * @return std::unique_ptr<Rectifier>
 */
std::unique_ptr<Rectifier> create_rectifier();

/**
 * @brief Create a rectifier object with a free function for the postprocess.
 *
 * @tparam F The type of the callback function.
 * @param fn Free function for the postprocess.
 * @return std::unique_ptr<Rectifier>
 */
template <
  typename F, std::enable_if_t<std::is_convertible_v<F, void (*)(const common::Image &)>, int> = 0>
inline std::unique_ptr<Rectifier> create_rectifier(F fn)
{
  auto processor = create_rectifier();
  auto fp = static_cast<void (*)(const common::Image &)>(fn);
  if (fp) processor->register_postprocess(fp);
  return processor;
}

/**
 * @brief Create a rectifier object with a member function for the postprocess.
 *
 * @tparam Obj The type of the object.
 * @tparam Method The type of the member function.
 * @param obj Pointer to the object.
 * @return std::unique_ptr<Rectifier>
 */
template <typename Obj, void (Obj::*Method)(const common::Image &)>
inline std::unique_ptr<Rectifier> create_rectifier(Obj * obj)
{
  auto processor = create_rectifier();
  processor->register_postprocess<Obj, Method>(obj);
  return processor;
}
}  // namespace accelerated_image_processor::pipeline
