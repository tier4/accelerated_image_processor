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
#include <accelerated_image_processor_common/parameter.hpp>
#include <accelerated_image_processor_common/processor.hpp>

#include <memory>
#include <optional>

namespace accelerated_image_processor::pipeline
{
class NppRectifier;
class OpenCvCudaRectifier;
class CpuRectifier;

/**
 * @brief Rectifier backend.
 */
enum class RectifierBackend : uint8_t { NPP, OPENCV_CUDA, CPU };

/**
 * @brief Abstract base class for rectifiers.
 */
class Rectifier : public common::BaseProcessor
{
public:
  explicit Rectifier(RectifierBackend backend, common::ParameterMap dedicated_parameters = {})
  : BaseProcessor(dedicated_parameters += {{"alpha", 0.0}}), backend_(backend)
  {
  }

  ~Rectifier() override = default;

  /**
   * @brief Return the scaling parameter alpha.
   */
  double alpha() { return this->parameter_value<double>("alpha"); }

  /**
   * @brief Return the rectification backend.
   */
  RectifierBackend backend() const { return backend_; }

  /**
   * @brief Set camera information before rectified, and compute the rectified camera information
   * under the hood.
   */
  void set_camera_info(const common::CameraInfo & camera_info)
  {
    camera_info_ = prepare_maps(camera_info);
  }

  /**
   * @brief Return the rectified camera information.
   */
  const common::CameraInfo & get_camera_info() const { return camera_info_.value(); }

  /**
   * @brief Return true if Rectifier::set_camera_info() was invoked and the rectified camera
   * information has been set.
   */
  bool is_ready() const { return camera_info_.has_value(); }

protected:
  /**
   * @brief Compute rectification maps and return the rectified camera information.
   * This function is called in Rectifier::set_camera_info() under the hood.
   */
  virtual common::CameraInfo prepare_maps(const common::CameraInfo & camera_info) = 0;

  std::optional<common::CameraInfo> camera_info_{std::nullopt};  //!< Rectified camera info.

private:
  const RectifierBackend backend_;  //!< Rectification backend type.
};

//!< @brief Factory function to create a NppRectifier.
std::unique_ptr<Rectifier> make_npp_rectifier();
//!< @brief Factory function to create a OpenCvCudaRectifier.
std::unique_ptr<Rectifier> make_opencv_cuda_rectifier();
//!< @brief Factory function to create a CpuRectifier.
std::unique_ptr<Rectifier> make_cpu_rectifier();
}  // namespace accelerated_image_processor::pipeline
