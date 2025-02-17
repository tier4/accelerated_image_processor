#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>

namespace gpu_imgproc {
namespace util {

class TaskQueue {
public:
    explicit TaskQueue(size_t queue_length=10): queue_length_(queue_length){};

    void addTask(std::function<void()> task) {
        std::unique_lock<std::mutex> lock(mutex_);
        tasks_.push(std::move(task));
        if (tasks_.size() > queue_length_) {
            // if the queue is too long, drop the oldest task
            tasks_.pop();
        }
        condition_.notify_one();
    }

    void run() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock, [this]{return request_stop_ || !tasks_.empty();});
                if (request_stop_ && tasks_.empty()) {
                    return;
                }
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }

    void stop() {
        std::unique_lock<std::mutex> lock(mutex_);
        request_stop_ = true;
        condition_.notify_one();
    }

protected:
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable condition_;
    bool request_stop_{false};
    size_t queue_length_;
};

}  // namespace util
}  // namespace gpu_imgproc
