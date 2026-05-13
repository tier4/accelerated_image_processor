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

#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage/serialized_bag_message.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <rosbag2_transport/reader_writer_factory.hpp>

#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace accelerated_image_processor::benchmark
{
class RosBagReader
{
public:
  /**
   * @brief Constructor for RosBagReader class.
   * @param bag_dir Directory path to the input rosbags.
   * @param storage_id Storage ID.
   */
  RosBagReader(const std::string & bag_dir, const std::string & storage_id)
  {
    rosbag2_storage::StorageOptions storage_options;
    storage_options.uri = bag_dir;
    storage_options.storage_id = storage_id;
    reader_ = rosbag2_transport::ReaderWriterFactory::make_reader(storage_options);
    reader_->open(storage_options);
  }

  /**
   * @brief Read messages from a specific topic.
   * @tparam T Type of message to read.
   * @param topic_name Name of the topic to read messages from.
   * @param max_count Maximum number of messages to read.
   * @return Vector of messages of type T.
   */
  template <typename T>
  std::vector<T> read_messages(
    const std::string & topic_name, size_t max_count = std::numeric_limits<size_t>::max())
  {
    std::vector<T> messages;
    rclcpp::Serialization<T> serialization;
    size_t count = 0;
    while (reader_->has_next() && count < max_count) {
      rosbag2_storage::SerializedBagMessageSharedPtr msg = reader_->read_next();

      if (msg->topic_name != topic_name) {
        continue;
      }

      rclcpp::SerializedMessage serialized(*msg->serialized_data);
      auto ros_msg = std::make_shared<T>();

      serialization.deserialize_message(&serialized, ros_msg.get());

      messages.push_back(*ros_msg);
      ++count;
    }
    return messages;
  }

private:
  std::unique_ptr<rosbag2_cpp::Reader> reader_;  //!< Pointer to the ROS bag reader.
};
}  // namespace accelerated_image_processor::benchmark
