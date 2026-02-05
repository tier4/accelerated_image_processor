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

#include "accelerated_image_processor_benchmark/benchmarker.hpp"

#include "accelerated_image_processor_benchmark/utility.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace accelerated_image_processor::benchmark
{
namespace
{
/**
 * @brief Calculate the percentile of a vector of values.
 *
 * @param values The vector of values.
 * @param p The percentile to calculate in [0.0, 100.0].
 * @return The percentile value. Returns NaN if the vector is empty.
 */
double percentile(const std::vector<double> & values, double p)
{
  if (values.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  std::vector<double> sorted_values(values.begin(), values.end());
  std::sort(sorted_values.begin(), sorted_values.end());

  const double clamped = std::min(100.0, std::max(0.0, p));
  const double index = clamped / 100.0 * (sorted_values.size() - 1);
  const size_t i0 = static_cast<size_t>(std::floor(index));
  const size_t i1 = static_cast<size_t>(std::ceil(index));

  if (i0 == i1) {
    return sorted_values[i0];
  }

  const double fraction = index - static_cast<double>(i0);

  return (1.0 - fraction) * sorted_values[i0] + fraction * sorted_values[i1];
}

/**
 * @brief Format bytes into a human-readable string.
 * @param bytes The number of bytes to format.
 * @return The formatted string.
 */
std::string format_bytes(uint64_t bytes)
{
  if (bytes < 1024) {
    return std::to_string(bytes) + "B";
  } else if (bytes < 1024 * 1024) {
    return std::to_string(bytes / 1024) + "KB";
  } else if (bytes < 1024 * 1024 * 1024) {
    return std::to_string(bytes / (1024 * 1024)) + "MB";
  } else {
    return std::to_string(bytes / (1024 * 1024 * 1024)) + "GB";
  }
}
}  // namespace

Benchmarker::Benchmarker(
  const YAML::Node & config, std::unique_ptr<common::BaseProcessor> processor)
: config_(config), processor_(std::move(processor))
{
  fetch_parameters(config_, processor_.get());
  print_processor(processor_.get());
  processor_->register_postprocess<Benchmarker, &Benchmarker::on_image>(this);
}

void Benchmarker::run(
  const std::vector<common::Image> & images, const size_t num_warmups, const size_t num_iterations)
{
  // Set source bytes
  this->set_source_bytes(images);

  auto try_processing = [this, &images](size_t iter_idx) {
    const auto idx = iter_idx % images.size();
    processor_->process(images[idx]);
  };

  // Warmup
  std::cout << ">>> Starting warmup [n = " << num_warmups << "]" << std::endl;
  for (size_t i = 0; i < num_warmups; ++i) {
    try_processing(i);
  }
  std::cout << "<<< ✨Finished warmup" << std::endl;

  // Iterations
  std::cout << ">>> Starting iterations [n = " << num_iterations << "]" << std::endl;
  this->reset_processed();
  for (size_t i = 0; i < num_iterations; ++i) {
    this->tic();
    try_processing(i);
  }
  std::cout << "<<< ✨Finished iterations" << std::endl;

  // Print benchmark results
  this->print();
}

void Benchmarker::on_image(const common::Image & image)
{
  iter_ms_.push_back(toc());
  processed_bytes_ += static_cast<uint64_t>(image.data.size());
  processed_count_++;
}

void Benchmarker::reset_processed()
{
  processed_bytes_ = 0;
  processed_count_ = 0;
  iter_ms_.clear();
  start_time_.reset();
}

void Benchmarker::tic()
{
  start_time_.emplace(std::chrono::steady_clock::now());
}

double Benchmarker::toc() const
{
  if (!start_time_) return 0.0;
  auto end_time = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end_time - start_time_.value()).count();
}

void Benchmarker::set_source_bytes(const std::vector<common::Image> & images)
{
  source_bytes_ = std::accumulate(
    images.begin(), images.end(), 0ull,
    [](uint64_t sum, const common::Image & img) { return sum + img.data.size(); });
}

double Benchmarker::compare_bytes() const
{
  return source_bytes_ > 0
           ? 100.0 * static_cast<double>(source_bytes_ - processed_bytes_) / source_bytes_
           : std::numeric_limits<double>::quiet_NaN();
}

double Benchmarker::total_ms() const
{
  return std::accumulate(iter_ms_.begin(), iter_ms_.end(), 0.0, std::plus<double>());
}

double Benchmarker::average_ms() const
{
  return processed_count_ > 0 ? total_ms() / processed_count_
                              : std::numeric_limits<double>::quiet_NaN();
}

double Benchmarker::percentile_ms(double p) const
{
  return percentile(iter_ms_, p);
}

double Benchmarker::fps() const
{
  return processed_count_ > 0 ? processed_count_ / total_ms()
                              : std::numeric_limits<double>::quiet_NaN();
}

void Benchmarker::print() const
{
  std::cout << "------------------ Benchmark Summary ------------------\n";

  std::cout << "Storage Ratio: " << compare_bytes() << "% (SOURCE=" << format_bytes(source_bytes_)
            << " -> PROCESSED=" << format_bytes(processed_bytes_) << ")\n";

  std::cout << "Iteration ms: [Average]=" << average_ms()  // Average
            << ", [50%tile]=" << percentile_ms(50)         // 50%tile
            << ", [90%tile]=" << percentile_ms(90)         // 90%tile
            << ", [FPS]=" << fps() << "\n";                // FPS

  std::cout << "-------------------------------------------------------\n";
}
}  // namespace accelerated_image_processor::benchmark
