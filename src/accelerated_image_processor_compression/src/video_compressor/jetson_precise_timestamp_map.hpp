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
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>

namespace accelerated_image_processor::compression
{
class TimestampMap
{
public:
  using PreciseTimestamp = int64_t;

private:
  // Helper class to pass nanosecond order timestamp.
  //
  // Because v4l2 buffers express its timestamp by the combination of tv_sec and tv_usec, which is
  // not sufficient to express nanosecond level ROS timestamp, use this class to maintain input
  // and output topic consistency
  using history_t = std::queue<PreciseTimestamp>;

public:
  explicit TimestampMap(const int v4l2_buffer_len)
  {
    // Allocate history (queue of timestamp) for each v4l2_buffer
    timestamp_history_ = std::make_unique<history_t[]>(v4l2_buffer_len);
  }

  void set(const uint32_t & buf_index, const PreciseTimestamp & ts)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    timestamp_history_[buf_index].push(ts);
  }

  bool get(const uint32_t & buf_index, PreciseTimestamp & ts)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (timestamp_history_[buf_index].empty()) {
      return false;
    }
    ts = timestamp_history_[buf_index].front();
    timestamp_history_[buf_index].pop();
    return true;
  }

private:
  std::unique_ptr<history_t[]> timestamp_history_;
  std::mutex mutex_;
};
}  // namespace accelerated_image_processor::compression
