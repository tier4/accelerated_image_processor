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

#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <ostream>
#include <vector>

namespace accelerated_image_processor::benchmark
{
/**
 * @brief Calculate the percentile of a vector of values.
 *
 * @param values The vector of values.
 * @param p The percentile to calculate in [0.0, 100.0].
 * @return The percentile value. Returns NaN if the vector is empty.
 */
inline double percentile(const std::vector<double> & values, double p)
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

class ProcessObserver
{
public:
  void on_image(const common::Image & image)
  {
    iter_ms_.push_back(toc());
    bytes_ += static_cast<uint64_t>(image.data.size());
    count_++;
  }

  void clear()
  {
    bytes_ = 0;
    count_ = 0;
    iter_ms_.clear();
    start_time_.reset();
  }

  void tic() { start_time_.emplace(std::chrono::steady_clock::now()); }

  double toc() const
  {
    if (!start_time_) return 0.0;
    auto end_time = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end_time - start_time_.value()).count();
  }

  double average_bytes() const
  {
    return count_ > 0 ? static_cast<double>(bytes_) / count_
                      : std::numeric_limits<double>::quiet_NaN();
  }

  double total_ms() const
  {
    return std::accumulate(iter_ms_.begin(), iter_ms_.end(), 0.0, std::plus<double>());
  }

  double average_ms() const
  {
    return count_ > 0 ? total_ms() / count_ : std::numeric_limits<double>::quiet_NaN();
  }

  double percentile_ms(double p) const { return percentile(iter_ms_, p); }

  double fps() const
  {
    return count_ > 0 ? count_ / total_ms() : std::numeric_limits<double>::quiet_NaN();
  }

  std::ostream & print_result(std::ostream & os) const
  {
    os << "Iteration ms: [Total]=" << total_ms() << ", [Average]=" << average_ms()
       << ", [50%tile]=" << percentile_ms(50) << ", [90%tile]=" << percentile_ms(90)
       << ", [FPS]=" << fps();
    return os;
  }

private:
  uint64_t bytes_ = 0;
  uint64_t count_ = 0;
  std::vector<double> iter_ms_;
  std::optional<std::chrono::steady_clock::time_point> start_time_;
};
}  // namespace accelerated_image_processor::benchmark
