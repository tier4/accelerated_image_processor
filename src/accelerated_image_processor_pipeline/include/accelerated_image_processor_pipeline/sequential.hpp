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
#include <accelerated_image_processor_common/processor.hpp>
#include <accelerated_image_processor_compression/builder.hpp>
#include <accelerated_image_processor_pipeline/builder.hpp>

#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace accelerated_image_processor::pipeline
{
/**
 * @brief Sequential processor that executes a sequence of processors in order.
 */
class Sequential final : public common::BaseProcessor
{
public:
  /**
   * @brief Child processor belongs to the sequence.
   */
  struct Child
  {
    Child(const std::string & ns, std::unique_ptr<common::BaseProcessor> processor)
    : ns(ns), processor(std::move(processor))
    {
    }

    std::string ns;                                    //!< Namespace of the processor
    std::unique_ptr<common::BaseProcessor> processor;  //!< Processor instance
  };

  Sequential() : common::BaseProcessor({}) {}

  // Delete copy constructor and assignment operator because this class holds processors as
  // unique_ptr.
  Sequential(const Sequential &) = delete;
  Sequential & operator=(const Sequential &) = delete;
  Sequential(Sequential &&) = default;
  Sequential & operator=(Sequential &&) = default;

  /**
   * @brief Append a processor to the sequence.
   *
   * @tparam P Processor type
   * @tparam Args Argument types
   *
   * @param ns Namespace of the processor
   * @param args Arguments to pass to the processor constructor
   * @return Sequential& Reference to the current instance
   */
  template <class P, class... Args>
  Sequential & append(const std::string & ns, Args &&... args)
  {
    if constexpr (std::is_same_v<P, Rectifier>) {
      sequence_.emplace_back(ns, create_rectifier(std::forward<Args>(args)...));
    } else if constexpr (std::is_same_v<P, compression::Compressor>) {
      sequence_.emplace_back(ns, compression::create_compressor(std::forward<Args>(args)...));
    }
    return *this;
  }

  /**
   * @brief Append a processor to the sequence with a callback function.
   *
   * @tparam P Processor type
   * @tparam F Callback function type
   * @tparam Args Argument types
   *
   * @param ns Namespace of the processor
   * @param fn Callback function to be called after processing
   * @param args Arguments to pass to the processor constructor
   * @return Sequential& Reference to the current instance
   */
  template <
    class P, class F,
    std::enable_if_t<std::is_convertible_v<F, void (*)(const common::Image &)>, int> = 0,
    class... Args>
  Sequential & append(const std::string & ns, F fn, Args &&... args)
  {
    if constexpr (std::is_same_v<P, Rectifier>) {
      sequence_.emplace_back(ns, create_rectifier<F>(std::forward<Args>(args)..., fn));
    } else if constexpr (std::is_same_v<P, compression::Compressor>) {
      sequence_.emplace_back(
        ns, compression::create_compressor<F>(std::forward<Args>(args)..., fn));
    }
    return *this;
  }

  /**
   * @brief Append a processor to the sequence with a callback function.
   *
   * @tparam P Processor type
   * @tparam Obj Object type
   * @tparam Method Method of the object to call
   * @tparam Args Argument types
   *
   * @param ns Namespace of the processor
   * @param obj Object instance to call the method on
   * @param args Arguments to pass to the processor constructor
   * @return Sequential& Reference to the current instance
   */
  template <class P, class Obj, void (Obj::*Method)(const common::Image &), class... Args>
  Sequential & append(const std::string & ns, Obj * obj, Args &&... args)
  {
    if constexpr (std::is_same_v<P, Rectifier>) {
      sequence_.emplace_back(ns, create_rectifier<Obj, Method>(std::forward<Args>(args)..., obj));
    } else if constexpr (std::is_same_v<P, compression::Compressor>) {
      sequence_.emplace_back(
        ns, compression::create_compressor<Obj, Method>(std::forward<Args>(args)..., obj));
    }
    return *this;
  }

  /**
   * @brief Set camera info for all processors in the pipeline.
   *
   * @note The camera info that is held by this class will be updated with the child's last valid
   * camera info.
   *
   * @param camera_info Camera info to set
   */
  void set_camera_info(const common::CameraInfo & camera_info) override;

  /**
   * @brief Return a writable reference to the sequence of child processors.
   *
   * @return std::vector<Child>& Reference to the sequence of child processors
   */
  std::vector<Child> & items() noexcept { return sequence_; }

  /**
   * @brief Return a read-only reference to the sequence of child processors.
   *
   * @return const std::vector<Child>& Reference to the sequence of child processors
   */
  const std::vector<Child> & items() const noexcept { return sequence_; }

private:
  /**
   * @brief Process an image sequentially through the pipeline.
   *
   * @param image Input image
   * @return common::Image Processed image
   */
  common::Image process_impl(const common::Image & image) override;

  std::vector<Child> sequence_;  //!< Sequence of child processors
};
}  // namespace accelerated_image_processor::pipeline
