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

#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace accelerated_image_processor::benchmark
{
inline bool iequals(std::string_view a, std::string_view b)
{
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    const auto ca = static_cast<unsigned char>(a[i]);
    const auto cb = static_cast<unsigned char>(b[i]);
    if (std::toupper(ca) != std::toupper(cb)) return false;
  }
  return true;
}

inline int to_int(const std::string & s, const char * name)
{
  char * end = nullptr;
  errno = 0;
  long v = std::strtol(s.c_str(), &end, 10);
  if (errno != 0 || end == s.c_str() || *end != '\0') {
    std::ostringstream oss;
    oss << "Invalid integer for " << name << ": '" << s << "'";
    throw std::runtime_error(oss.str());
  }
  if (v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max()) {
    std::ostringstream oss;
    oss << "Out of range integer for " << name << ": '" << s << "'";
    throw std::runtime_error(oss.str());
  }
  return static_cast<int>(v);
}

inline uint64_t to_u64(const std::string & s, const char * name)
{
  char * end = nullptr;
  errno = 0;
  unsigned long long v = std::strtoull(s.c_str(), &end, 10);
  if (errno != 0 || end == s.c_str() || *end != '\0') {
    std::ostringstream oss;
    oss << "Invalid uint64 for " << name << ": '" << s << "'";
    throw std::runtime_error(oss.str());
  }
  return static_cast<uint64_t>(v);
}
}  // namespace accelerated_image_processor::benchmark
