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

namespace accelerated_image_processor::benchmark
{
/**
 * @brief Create a synthetic image with the given height, width, encoding, seed, and index.
 *
 * @param height The height of the image.
 * @param width The width of the image.
 * @param encoding The encoding of the image.
 * @param seed The seed for the random number generator.
 * @param index The index of the image.
 * @return common::Image The synthetic image.
 */
common::Image make_synthetic_image(
  int height, int width, common::ImageEncoding encoding, uint64_t seed, uint64_t index);
}  // namespace accelerated_image_processor::benchmark
