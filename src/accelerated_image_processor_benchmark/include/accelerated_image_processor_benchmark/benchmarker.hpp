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

#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace accelerated_image_processor::benchmark
{
class Benchmarker
{
public:
  Benchmarker(const YAML::Node & config, std::unique_ptr<common::BaseProcessor> processor);

  void run(
    const std::vector<common::Image> & images, const size_t num_warmups,
    const size_t num_iterations);

  /**
   * @brief Process an image for benchmarking.
   * @param image The image to process.
   */
  void on_image(const common::Image & image);

  /**
   * @brief Reset the processed count.
   */
  void reset_processed();

  /**
   * @brief Start the benchmark timer.
   */
  void tic();

  /**
   * @brief Stop the benchmark timer.
   */
  double toc() const;

  /**
   * @brief Set the source bytes for benchmarking.
   * @param images The images to process.
   */
  void set_source_bytes(const std::vector<common::Image> & images);

  /**
   * @brief Compare the processed bytes with the source bytes.
   */
  double compare_bytes() const;

  /**
   * @brief Get the total time in milliseconds.
   */
  double total_ms() const;

  /**
   * @brief Get the average time per iteration in milliseconds.
   */
  double average_ms() const;

  /**
   * @brief Get the percentile time in milliseconds.
   */
  double percentile_ms(double p) const;

  /**
   * @brief Get the frames per second.
   */
  double fps() const;

  /**
   * @brief Print the benchmark results.
   */
  void print() const;

private:
  YAML::Node config_;
  std::unique_ptr<common::BaseProcessor> processor_;

  uint64_t source_bytes_ = 0;
  uint64_t processed_bytes_ = 0;
  uint64_t processed_count_ = 0;
  std::vector<double> iter_ms_;
  std::optional<std::chrono::steady_clock::time_point> start_time_;
};
}  // namespace accelerated_image_processor::benchmark
