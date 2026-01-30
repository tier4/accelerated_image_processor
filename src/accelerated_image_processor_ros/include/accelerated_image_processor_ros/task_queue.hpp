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
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>

namespace accelerated_image_processor::ros
{
class TaskQueue
{
public:
  explicit TaskQueue(size_t queue_size = 10) : queue_size_(queue_size) {}

  void add_task(std::function<void()> && task)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    tasks_.push(std::move(task));
    if (tasks_.size() > queue_size_) {
      tasks_.pop();
    }
    condition_.notify_one();
  }

  void run()
  {
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return request_stop_ || !tasks_.empty(); });
        if (request_stop_ && tasks_.empty()) {
          return;
        }
        task = std::move(tasks_.front());
        tasks_.pop();
      }
      task();
    }
  }

  void stop()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    request_stop_ = true;
    condition_.notify_one();
  }

private:
  size_t queue_size_;
  std::queue<std::function<void()>> tasks_;
  std::mutex mutex_;
  std::condition_variable condition_;
  bool request_stop_{false};
};

class TaskWorker
{
public:
  explicit TaskWorker(size_t max_queue_size)
  : queue_(max_queue_size), thread_(&TaskQueue::run, &queue_)
  {
  }

  template <typename F>
  void add_task(F && task)
  {
    queue_.add_task(std::forward<F>(task));
  }

  ~TaskWorker()
  {
    queue_.stop();
    thread_.join();
  }

private:
  TaskQueue queue_;
  std::thread thread_;
};
}  // namespace accelerated_image_processor::ros
