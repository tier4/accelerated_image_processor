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

#include <accelerated_image_processor_common/processor.hpp>
#include <accelerated_image_processor_compression/compressor.hpp>

#include <memory>
#include <string>
#include <type_traits>

namespace accelerated_image_processor::compression
{
/**
 * @brief Convert a string to a compression type
 * @param str String to convert expected strings ["JPEG", "VIDEO"]
 * @return CompressionType
 */
CompressionType to_compression_type(const std::string & str);

/**
 * @brief Create a compressor processor.
 *
 * @param type Compression type
 * @return std::unique_ptr<Compressor>
 */
std::unique_ptr<Compressor> create_compressor(CompressionType type);

/**
 * @brief Create a compressor processor.
 *
 * @param type Compression type name in string format
 * @return std::unique_ptr<Compressor>
 */
std::unique_ptr<Compressor> create_compressor(const std::string & type);

/**
 * @brief Create a compressor processor with a free function for the postprocess.
 *
 * @param type Compression type
 * @param fn Free function for the postprocess
 * @return std::unique_ptr<Compressor>
 */
template <
  typename F, std::enable_if_t<std::is_convertible_v<F, void (*)(const common::Image &)>, int> = 0>
inline std::unique_ptr<Compressor> create_compressor(CompressionType type, F fn)
{
  auto processor = create_compressor(type);
  auto fp = static_cast<void (*)(const common::Image &)>(fn);
  if (fp) processor->register_postprocess(fp);
  return processor;
}

/**
 * @brief Create a compressor processor with a free function for the postprocess.
 *
 * @param type Compression type name in string format
 * @param fn Free function for the postprocess
 * @return std::unique_ptr<Compressor>
 */
template <
  typename F, std::enable_if_t<std::is_convertible_v<F, void (*)(const common::Image &)>, int> = 0>
inline std::unique_ptr<Compressor> create_compressor(const std::string & type, F fn)
{
  return create_compressor(to_compression_type(type), fn);
}

/**
 * @brief Create a compressor processor with a member function for the postprocess.
 *
 * @param type Compression type
 * @param obj Object that has a member function for the postprocess
 * @return std::unique_ptr<Compressor>
 */
template <typename Obj, void (Obj::*Method)(const common::Image &)>
inline std::unique_ptr<Compressor> create_compressor(CompressionType type, Obj * obj)
{
  auto processor = create_compressor(type);
  processor->register_postprocess<Obj, Method>(obj);
  return processor;
}

/**
 * @brief Create a compressor processor with a member function for the postprocess.
 *
 * @param type Compression type name in string format
 * @param obj Object that has a member function for the postprocess
 * @return std::unique_ptr<Compressor>
 */
template <typename Obj, void (Obj::*Method)(const common::Image &)>
inline std::unique_ptr<Compressor> create_compressor(const std::string & type, Obj * obj)
{
  return create_compressor<Obj, Method>(to_compression_type(type), obj);
}
}  // namespace accelerated_image_processor::compression
