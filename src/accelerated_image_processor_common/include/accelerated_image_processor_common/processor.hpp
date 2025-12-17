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

#include "accelerated_image_processor_common/datatype.hpp"
#include "accelerated_image_processor_common/parameter.hpp"

#include <functional>
#include <optional>
#include <string>
#include <utility>

namespace accelerated_image_processor::common
{
/**
 * @brief Base class for image processors.
 */
class BaseProcessor
{
public:
  // @brief Post-processing function type.
  using PostProcessFn = std::function<void(common::Image &)>;

  /**
   * @brief Constructor.
   */
  explicit BaseProcessor(ParameterMap parameters) : parameters_(std::move(parameters)) {}

  virtual ~BaseProcessor() = default;

  /**
   * @brief Set the post-processing function and return a reference to the current object.
   * @param postprocess_fn The post-processing function.
   */
  void with_postprocess(const PostProcessFn & postprocess_fn) { postprocess_fn_ = postprocess_fn; }

  /**
   * @brief Process the input image.
   * @param image The input image.
   */
  void process(common::Image & image)
  {
    auto processed = this->process_impl(image);
    if (postprocess_fn_) {
      postprocess_fn_.value()(processed);
    }
  };

  /**
   * @brief Return the read/write reference to the parameters.
   */
  ParameterMap & parameters() { return parameters_; }

  /**
   * @brief Return the read-only reference to the parameters.
   */
  const ParameterMap & parameters() const { return parameters_; }

  /**
   * @brief Return the read/write reference to the parameter value.
   * @tparam T The type of the parameter value.
   * @param key The key of the parameter.
   * @return The value of the parameter.
   */
  template <typename T>
  T & parameter_value(const std::string & key)
  {
    return std::get<T>(parameters_.at(key));
  }

  /**
   * @brief Return the read-only reference to the parameter value.
   * @tparam T The type of the parameter value.
   * @param key The key of the parameter.
   * @return The value of the parameter.
   */
  template <typename T>
  const T & parameter_value(const std::string & key) const
  {
    return std::get<T>(parameters_.at(key));
  }

protected:
  /**
   * @brief Process the input image.
   * @param image The input image.
   * @return The processed image.
   */
  virtual common::Image process_impl(const common::Image & image) = 0;

protected:
  ParameterMap parameters_;                      // Parameters for the processor.
  std::optional<PostProcessFn> postprocess_fn_;  //!< Optional post-processing function.
};
}  // namespace accelerated_image_processor::common
