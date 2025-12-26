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

  void set_camera_info(const common::CameraInfo & camera_info)
  {
    camera_info_ = prepare_maps(camera_info);
  }

  const common::CameraInfo & get_camera_info() const { return camera_info_.value(); }

  double alpha() { return this->parameter_value<double>("alpha"); }

  bool is_ready() { return camera_info_.has_value(); }

  RectifierBackend backend() const { return backend_; }

protected:
  virtual common::CameraInfo prepare_maps(const common::CameraInfo & camera_info) = 0;

  std::optional<common::CameraInfo> camera_info_{std::nullopt};  //!< Camera information.

private:
  const RectifierBackend backend_;
};

std::unique_ptr<Rectifier> make_npp_rectifier();
std::unique_ptr<Rectifier> make_opencv_cuda_rectifier();
std::unique_ptr<Rectifier> make_cpu_rectifier();
}  // namespace accelerated_image_processor::pipeline
