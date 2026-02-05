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

#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace accelerated_image_processor::common
{
/**
 * @brief Base class for image processors.
 */
class BaseProcessor
{
private:
  /**
   * @brief A holder to wrap a invoking free function.
   */
  struct FreeFn
  {
    using invoker_t = void (*)(const Image &);
    invoker_t invoker = nullptr;
  };

  /**
   * @brief A holder to wrap a invoking member function.
   */
  struct MemberFn
  {
    using invoker_t = void (*)(void *, const Image &);
    void * obj{};
    invoker_t invoker = nullptr;
  };

  using FnStorage = std::variant<std::monostate, FreeFn, MemberFn>;

  FnStorage storage_{std::monostate{}};  //!< Storage of the postprocess function.
  ParameterMap parameters_;              //!< Parameters for the processor.

  /**
   * @brief Restore the member pointer from method_bits and invoke its method.
   */
  template <typename Obj, void (Obj::*Method)(const Image &)>
  static void invoke_member(void * p, const Image & img)
  {
    auto * o = static_cast<Obj *>(p);
    if (o) (o->*Method)(img);
  }

public:
  /**
   * @brief Constructor.
   */
  explicit BaseProcessor(ParameterMap parameters) : parameters_(std::move(parameters)) {}

  virtual ~BaseProcessor() = default;

  /**
   * @brief Register a free function as the postprocess function.
   * @param fn Free function for the postprocess function.
   */
  template <typename F, std::enable_if_t<std::is_convertible_v<F, FreeFn::invoker_t>, int> = 0>
  void register_postprocess(F fn)
  {
    storage_ = FreeFn{static_cast<FreeFn::invoker_t>(fn)};
  }

  /**
   * @brief Register a member function as the postprocess function.
   */
  template <typename Obj, void (Obj::*Method)(const Image &)>
  void register_postprocess(Obj * obj)
  {
    storage_ = MemberFn{obj, &invoke_member<Obj, Method>};
  }

  /**
   * @brief Process the input image.
   * @param image The input image.
   * @return The processed image if the process is successful, otherwise std::nullopt.
   */
  virtual std::optional<common::Image> process(const Image & image)
  {
    if (!is_ready()) {
      // TODO(ktro2828): Update to return a type that describes if the process success or not
      // instead of void
      return std::nullopt;
    }

    auto processed = this->process_impl(image);

    if (!processed.is_valid()) {
      // TODO(ktro2828): Update to return a type that describes if the process success or not
      // instead of void
      return std::nullopt;
    }

    post_process(processed);

    return processed;
  };

  /**
   * @brief Execute post process
   * @param processed The image to be post-processed
   */
  void post_process(Image & processed)
  {
    std::visit(
      [&processed](auto & f) {
        using T = std::decay_t<decltype(f)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          // do nothing
        } else if constexpr (std::is_same_v<T, FreeFn>) {
          if (f.invoker) f.invoker(processed);
        } else if constexpr (std::is_same_v<T, MemberFn>) {
          if (f.invoker && f.obj) f.invoker(f.obj, processed);
        }
      },
      storage_);
  }

  /**
   * @brief Check the processor is ready to run processing.
   */
  virtual bool is_ready() const { return true; }

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
  virtual Image process_impl(const Image & image) = 0;
};
}  // namespace accelerated_image_processor::common
