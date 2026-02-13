// Copyright 2026 TIER IV, Inc.
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

#include <accelerated_image_processor_common/processor.hpp>

#include <memory>
#include <string>
#include <type_traits>

namespace accelerated_image_processor::decompression
{
/**
 * @brief Type alias for decompressor processor.
 * @note This might be a specific implementation of a decompressor processor in the future.
 */
using Decompressor = common::BaseProcessor;

/**
 * @brief Compression type enum
 */
enum class DecompressionType : uint8_t { VIDEO };

/**
 * @brief Convert a string to a decompression type
 * @param str String to convert expected strings ["VIDEO"]
 * @return DecompressionType
 */
DecompressionType to_decompression_type(const std::string & str);

/**
 * @brief Create a decompressor processor.
 *
 * @param type Decompression type
 * @return std::unique_ptr<Decompressor>
 */
std::unique_ptr<Decompressor> create_decompressor(DecompressionType type);

/**
 * @brief Create a decompressor processor.
 *
 * @param type Decompression type name in string format
 * @return std::unique_ptr<Decompressor>
 */
std::unique_ptr<Decompressor> create_decompressor(const std::string & type);

/**
 * @brief Create a decompressor processor with a free function for the postprocess.
 *
 * @param type Decompression type
 * @param fn Free function for the postprocess
 * @return std::unique_ptr<Decompressor>
 */
template <
  typename F, std::enable_if_t<std::is_convertible_v<F, void (*)(const common::Image &)>, int> = 0>
inline std::unique_ptr<Decompressor> create_decompressor(DecompressionType type, F fn)
{
  auto processor = create_decompressor(type);
  auto fp = static_cast<void (*)(const common::Image &)>(fn);
  if (fp) processor->register_postprocess(fp);
  return processor;
}

/**
 * @brief Create a decompressor processor with a free function for the postprocess.
 *
 * @param type Decompression type name in string format
 * @param fn Free function for the postprocess
 * @return std::unique_ptr<Decompressor>
 */
template <
  typename F, std::enable_if_t<std::is_convertible_v<F, void (*)(const common::Image &)>, int> = 0>
inline std::unique_ptr<Decompressor> create_decompressor(const std::string & type, F fn)
{
  return create_decompressor(to_decompression_type(type), fn);
}

/**
 * @brief Create a decompressor processor with a member function for the postprocess.
 *
 * @param type Decompression type
 * @param obj Object that has a member function for the postprocess
 * @return std::unique_ptr<Decompressor>
 */
template <typename Obj, void (Obj::*Method)(const common::Image &)>
inline std::unique_ptr<Decompressor> create_decompressor(DecompressionType type, Obj * obj)
{
  auto processor = create_decompressor(type);
  processor->register_postprocess<Obj, Method>(obj);
  return processor;
}

/**
 * @brief Create a decompressor processor with a member function for the postprocess.
 *
 * @param type Decompression type name in string format
 * @param obj Object that has a member function for the postprocess
 * @return std::unique_ptr<Decompressor>
 */
template <typename Obj, void (Obj::*Method)(const common::Image &)>
inline std::unique_ptr<Decompressor> create_decompressor(const std::string & type, Obj * obj)
{
  return create_decompressor<Obj, Method>(to_decompression_type(type), obj);
}
}  // namespace accelerated_image_processor::decompression
