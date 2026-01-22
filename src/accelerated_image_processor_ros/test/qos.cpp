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

#include "accelerated_image_processor_ros/qos.hpp"

#include <std_msgs/msg/string.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>

using std::chrono_literals::operator""s;
using std::chrono_literals::operator""ms;

namespace accelerated_image_processor::ros
{
namespace
{
struct RosGuard
{
  RosGuard(int & argc, char **& argv) { rclcpp::init(argc, argv); }
  ~RosGuard()
  {
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
  }
};

// Spin the given node for up to `timeout`, yielding periodically.
void spin_for(const std::shared_ptr<rclcpp::Node> & node, std::chrono::milliseconds timeout)
{
  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node);

  const auto start = std::chrono::steady_clock::now();
  while (rclcpp::ok() && (std::chrono::steady_clock::now() - start) < timeout) {
    exec.spin_some();
    std::this_thread::sleep_for(2ms);
  }

  exec.remove_node(node);
}

// Spin two nodes together for up to `timeout` to allow graph discovery.
void spin_for(
  const std::shared_ptr<rclcpp::Node> & a, const std::shared_ptr<rclcpp::Node> & b,
  std::chrono::milliseconds timeout)
{
  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(a);
  exec.add_node(b);

  const auto start = std::chrono::steady_clock::now();
  while (rclcpp::ok() && (std::chrono::steady_clock::now() - start) < timeout) {
    exec.spin_some();
    std::this_thread::sleep_for(2ms);
  }

  exec.remove_node(a);
  exec.remove_node(b);
}

// Poll find_qos until it returns a value or timeout expires.
std::optional<rclcpp::QoS> wait_find_qos(
  const std::shared_ptr<rclcpp::Node> & node, const std::string & topic,
  std::chrono::milliseconds timeout)
{
  const auto start = std::chrono::steady_clock::now();
  while (rclcpp::ok() && (std::chrono::steady_clock::now() - start) < timeout) {
    // Allow graph/cache updates.
    spin_for(node, 10ms);

    auto qos = find_qos(node.get(), topic);
    if (qos.has_value()) {
      return qos;
    }

    std::this_thread::sleep_for(5ms);
  }
  return std::nullopt;
}
}  // namespace

// ----------------------------
// Unit-style behavior tests
// ----------------------------

TEST(TestQoSFindQos, ReturnsNulloptWhenNoPublishers)
{
  auto node = std::make_shared<rclcpp::Node>("aip_qos_unit_no_publishers");

  const auto qos = find_qos(node.get(), "/aip_qos_unit/no_publishers");
  EXPECT_FALSE(qos.has_value());
}

TEST(TestQoSFindQos, ReturnsNulloptWhenMultiplePublishers)
{
  auto node = std::make_shared<rclcpp::Node>("aip_qos_unit_multiple_publishers");

  // Create two publishers on the same node for the same topic.
  // This should cause get_publishers_info_by_topic() to return >1 entries.
  const std::string topic = "/aip_qos_unit/multiple_publishers";
  (void)node->create_publisher<std_msgs::msg::String>(topic, rclcpp::QoS(10));
  (void)node->create_publisher<std_msgs::msg::String>(topic, rclcpp::QoS(10));

  // Give the graph some time to register publishers before querying.
  spin_for(node, 200ms);

  const auto qos = find_qos(node.get(), topic);
  EXPECT_FALSE(qos.has_value());
}

// ----------------------------
// Integration-style behavior tests
// ----------------------------

TEST(TestQoSFindQos, SucceedsWithSinglePublisherAndReturnsProfile)
{
  auto pub_node = std::make_shared<rclcpp::Node>("aip_qos_it_pub");
  auto query_node = std::make_shared<rclcpp::Node>("aip_qos_it_query");

  const std::string topic = "/aip_qos_it/single_publisher";

  // Use a non-default profile to make it more likely we can sanity-check something meaningful.
  // Depth should be reflected in the profile's history depth.
  rclcpp::QoS pub_qos(rclcpp::KeepLast(7));
  pub_qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  pub_qos.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);

  auto pub = pub_node->create_publisher<std_msgs::msg::String>(topic, pub_qos);
  (void)pub;

  // Spin both nodes so discovery happens.
  spin_for(pub_node, query_node, 500ms);

  auto found = wait_find_qos(query_node, topic, 2s);
  ASSERT_TRUE(found.has_value());

  // Basic assertions. QoS equality isn't directly comparable, so we check key fields.
  const rmw_qos_profile_t profile = found->get_rmw_qos_profile();
  EXPECT_EQ(profile.history, RMW_QOS_POLICY_HISTORY_KEEP_LAST);
  EXPECT_EQ(profile.depth, 7u);
  EXPECT_EQ(profile.reliability, RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  EXPECT_EQ(profile.durability, RMW_QOS_POLICY_DURABILITY_VOLATILE);
}

TEST(TestQoSFindQos, FailsWhenTwoDifferentNodesPublishSameTopic)
{
  auto pub_node1 = std::make_shared<rclcpp::Node>("aip_qos_it_pub1");
  auto pub_node2 = std::make_shared<rclcpp::Node>("aip_qos_it_pub2");
  auto query_node = std::make_shared<rclcpp::Node>("aip_qos_it_query2");

  const std::string topic = "/aip_qos_it/two_publishers";

  auto pub1 = pub_node1->create_publisher<std_msgs::msg::String>(topic, rclcpp::QoS(1));
  auto pub2 = pub_node2->create_publisher<std_msgs::msg::String>(topic, rclcpp::QoS(5));
  (void)pub1;
  (void)pub2;

  // Spin all nodes to ensure the graph sees both publishers.
  {
    rclcpp::executors::SingleThreadedExecutor exec;
    exec.add_node(pub_node1);
    exec.add_node(pub_node2);
    exec.add_node(query_node);
    const auto start = std::chrono::steady_clock::now();
    while (rclcpp::ok() && (std::chrono::steady_clock::now() - start) < 800ms) {
      exec.spin_some();
      std::this_thread::sleep_for(2ms);
    }
    exec.remove_node(pub_node1);
    exec.remove_node(pub_node2);
    exec.remove_node(query_node);
  }

  const auto qos = find_qos(query_node.get(), topic);

  EXPECT_FALSE(qos.has_value());
}
}  // namespace accelerated_image_processor::ros

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  accelerated_image_processor::ros::RosGuard guard(argc, argv);
  return RUN_ALL_TESTS();
}
