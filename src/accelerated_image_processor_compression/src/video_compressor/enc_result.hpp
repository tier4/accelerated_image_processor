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
#include <string>

namespace accelerated_image_processor::compression
{
/**
 * @brief Holds the raw result of an encoder API call and readable message.
 */
struct EncStatus
{
  bool ok{true};        // true if the encoder API call succeeded
  std::string message;  // error text when !ok

  EncStatus() = default;
  explicit EncStatus(bool r, const std::string & m = "") : ok(r), message(m) {}
};

/**
 * @brief Result of a (possibly multi-step) encoder operation.
 */
struct EncResult
{
  bool ok{true};
  EncStatus status;

  EncResult() = default;
  explicit EncResult(const EncStatus & s) : ok(s.ok), status(s) {}
  explicit EncResult(bool o, const EncStatus & s) : ok(o), status(s) {}
  static EncResult success() { return EncResult(EncStatus{true, ""}); }
};
}  // namespace accelerated_image_processor::compression
