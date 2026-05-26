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
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <ostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace accelerated_image_processor::benchmark
{
namespace
{
constexpr auto processing_timeout = std::chrono::seconds(30);

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
  const std::vector<common::Image> & images, const size_t num_warmups, const size_t num_iterations,
  const std::optional<float> frame_rate)
{
  // Set source bytes
  this->set_source_bytes(images, num_iterations);

  auto try_processing = [this, &images](size_t iter_idx) {
    const auto idx = iter_idx % images.size();
    this->tic(images[idx]);
    processor_->process(images[idx]);
  };

  // Warmup
  std::cout << ">>> Starting warmup [n = " << num_warmups << "]" << std::endl;
  this->reset_processed();
  for (size_t i = 0; i < num_warmups; ++i) {
    try_processing(i);
  }
  this->wait_for_processed(num_warmups);
  std::cout << "<<< ✨Finished warmup" << std::endl;

  // Iterations
  std::cout << ">>> Starting iterations [n = " << num_iterations << "]" << std::endl;
  std::chrono::duration<float> sleep_sec;
  if (frame_rate) {
    sleep_sec = std::chrono::duration<float>(1.0f / frame_rate.value());
  }
  this->reset_processed();
  for (size_t i = 0; i < num_iterations; ++i) {
    if (frame_rate) {
      std::this_thread::sleep_for(sleep_sec);
    }
    try_processing(i);
  }
  this->wait_for_processed(num_iterations);
  std::cout << "<<< ✨Finished iterations" << std::endl;

  // Print benchmark results
  this->print();
}

void Benchmarker::on_image(const common::Image & image)
{
  const auto elapsed_ms = toc(image);

  std::lock_guard<std::mutex> lock(mutex_);
  if (elapsed_ms) {
    iter_ms_.push_back(elapsed_ms.value());
    processed_bytes_ += static_cast<uint64_t>(image.data.size());
    processed_count_++;
    benchmark_end_time_ = std::chrono::steady_clock::now();
  } else {
    unmatched_count_++;
  }
  processed_cv_.notify_all();
}

void Benchmarker::reset_processed()
{
  std::lock_guard<std::mutex> lock(mutex_);
  processed_bytes_ = 0;
  processed_count_ = 0;
  unmatched_count_ = 0;
  iter_ms_.clear();
  start_times_.clear();
  benchmark_start_time_.reset();
  benchmark_end_time_.reset();
}

void Benchmarker::tic(const common::Image & image)
{
  std::lock_guard<std::mutex> lock(mutex_);
  const auto now = std::chrono::steady_clock::now();
  start_times_[image.timestamp].push_back(now);
  if (!benchmark_start_time_) {
    benchmark_start_time_ = now;
  }
}

std::optional<double> Benchmarker::toc(const common::Image & image)
{
  const auto end_time = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = start_times_.find(image.timestamp);
  if (it == start_times_.end() || it->second.empty()) {
    return std::nullopt;
  }

  const auto start_time = it->second.front();
  it->second.pop_front();
  if (it->second.empty()) {
    start_times_.erase(it);
  }

  return std::chrono::duration<double, std::milli>(end_time - start_time).count();
}

bool Benchmarker::wait_for_processed(size_t expected_count)
{
  std::unique_lock<std::mutex> lock(mutex_);
  const bool completed = processed_cv_.wait_for(lock, processing_timeout, [this, expected_count] {
    return processed_count_ >= expected_count;
  });

  if (!completed) {
    std::cout << "Warning: timed out waiting for processed images (" << processed_count_ << "/"
              << expected_count << ")\n";
  }

  return completed;
}

void Benchmarker::set_source_bytes(
  const std::vector<common::Image> & images, const size_t num_iterations)
{
  uint64_t source_bytes = 0;
  for (size_t i = 0; i < num_iterations; ++i) {
    source_bytes += static_cast<uint64_t>(images[i % images.size()].data.size());
  }

  std::lock_guard<std::mutex> lock(mutex_);
  source_bytes_ = source_bytes;
}

double Benchmarker::compare_bytes() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return source_bytes_ > 0
           ? 100.0 * static_cast<double>(source_bytes_ - processed_bytes_) / source_bytes_
           : std::numeric_limits<double>::quiet_NaN();
}

double Benchmarker::total_ms() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return std::accumulate(iter_ms_.begin(), iter_ms_.end(), 0.0, std::plus<double>());
}

double Benchmarker::average_ms() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  const double total_ms =
    std::accumulate(iter_ms_.begin(), iter_ms_.end(), 0.0, std::plus<double>());
  return processed_count_ > 0 ? total_ms / processed_count_
                              : std::numeric_limits<double>::quiet_NaN();
}

double Benchmarker::percentile_ms(double p) const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return percentile(iter_ms_, p);
}

double Benchmarker::fps() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (!benchmark_start_time_ || !benchmark_end_time_ || processed_count_ == 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  const auto wall_ms = std::chrono::duration<double, std::milli>(
                         benchmark_end_time_.value() - benchmark_start_time_.value())
                         .count();
  return wall_ms > 0.0 ? 1000.0 * processed_count_ / wall_ms
                       : std::numeric_limits<double>::quiet_NaN();
}

void Benchmarker::print() const
{
  std::vector<double> iter_ms;
  uint64_t source_bytes = 0;
  uint64_t processed_bytes = 0;
  uint64_t processed_count = 0;
  uint64_t unmatched_count = 0;
  double wall_ms = std::numeric_limits<double>::quiet_NaN();

  {
    std::lock_guard<std::mutex> lock(mutex_);
    iter_ms = iter_ms_;
    source_bytes = source_bytes_;
    processed_bytes = processed_bytes_;
    processed_count = processed_count_;
    unmatched_count = unmatched_count_;
    if (benchmark_start_time_ && benchmark_end_time_) {
      wall_ms = std::chrono::duration<double, std::milli>(
                  benchmark_end_time_.value() - benchmark_start_time_.value())
                  .count();
    }
  }

  const double total_ms = std::accumulate(iter_ms.begin(), iter_ms.end(), 0.0, std::plus<double>());
  const double average_ms =
    processed_count > 0 ? total_ms / processed_count : std::numeric_limits<double>::quiet_NaN();
  const double fps =
    wall_ms > 0.0 ? 1000.0 * processed_count / wall_ms : std::numeric_limits<double>::quiet_NaN();
  const double storage_reduction =
    source_bytes > 0 ? 100.0 * static_cast<double>(source_bytes - processed_bytes) / source_bytes
                     : std::numeric_limits<double>::quiet_NaN();

  std::cout << "------------------ Benchmark Summary ------------------\n";

  std::cout << "Storage Reduction: " << storage_reduction << "% "  // Reduction rate[%]
            << "(" << format_bytes(source_bytes) << " -> "         // Source size
            << format_bytes(processed_bytes) << ")\n";             // Processed size

  std::cout << "Latency ms: [Average]=" << average_ms     // Average
            << ", [50%tile]=" << percentile(iter_ms, 50)  // 50%tile
            << ", [90%tile]=" << percentile(iter_ms, 90)  // 90%tile
            << ", [FPS]=" << fps << "\n";                 // FPS

  if (unmatched_count > 0) {
    std::cout << "Warning: " << unmatched_count
              << " output frames were not matched to benchmark inputs.\n";
  }

  std::cout << "-------------------------------------------------------\n";
}
}  // namespace accelerated_image_processor::benchmark
