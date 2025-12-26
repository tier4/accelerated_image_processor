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

#include <accelerated_image_processor_common/parameter.hpp>
#include <accelerated_image_processor_common/processor.hpp>
#include <opencv2/core.hpp>

#include <array>
#include <optional>
#include <vector>

#ifdef NPP_AVAILABLE
#include <cuda_runtime.h>
#include <nppdefs.h>
#endif

#ifdef OPENCV_CUDA_AVAILABLE
#include <opencv2/core/cuda.hpp>
#endif

namespace accelerated_image_processor::pipeline
{
/**
 * @brief Camera information.
 * @todo Implement CameraInfo to common package.
 */
struct CameraInfo
{
  size_t height;             //!< Height of the image in pixels.
  size_t width;              //!< Width of the image in pixels.
  std::vector<double> d;     //!< Distortion coefficients.
  std::array<double, 9> k;   //!< Intrinsic camera matrix.
  std::array<double, 12> p;  //!< Extrinsic camera matrix.
};

/**
 * @brief Abstract base class for rectifiers.
 */
class Rectifier : public common::BaseProcessor
{
public:
  explicit Rectifier(common::ParameterMap dedicated_parameters = {})
  : BaseProcessor(dedicated_parameters += {{"alpha", 0.0}})
  {
  }

  ~Rectifier() override = default;

  void set_camera_info(const CameraInfo & camera_info) { camera_info_ = prepare_maps(camera_info); }

  const CameraInfo & get_camera_info() const { return camera_info_.value(); }

  double alpha() { return this->parameter_value<double>("alpha"); }

  bool is_ready() { return camera_info_.has_value(); }

protected:
  virtual CameraInfo prepare_maps(const CameraInfo & camera_info) = 0;

  std::optional<CameraInfo> camera_info_{std::nullopt};  //!< Camera information.
};

#ifdef NPP_AVAILABLE
/**
 * @brief Rectifier using NPP.
 */
class NPPRectifier final : public Rectifier
{
public:
  NPPRectifier();
  ~NPPRectifier() override;

private:
  common::Image process_impl(const common::Image & image) override;

  Npp32f * map_x_{nullptr};
  Npp32f * map_y_{nullptr};
  int map_x_step_;
  int map_y_step_;
  Npp8u * src_{nullptr};
  Npp8u * dst_{nullptr};
  int src_step_;
  int dst_step_;
  cudaStream_t stream_;
};
#endif  // NPP_AVAILABLE

#ifdef OPENCV_CUDA_AVAIL
/**
 * @brief Rectifier using OpenCV CUDA.
 */
class OpenCVCUDARectifier final : public Rectifier
{
public:
  OpenCVCUDARectifier();
  ~OpenCVCUDARectifier() override = default;

private:
  common::Image process_impl(const common::Image & image) override;

  CameraInfo prepare_maps(const CameraInfo & camera_info) override;

  cv::cuda::GpuMat map_x_;
  cv::cuda::GpuMat map_y_;
};
#endif  // OPENCV_CUDA_AVAILABLE

/**
 * @brief Rectifier using OpenCV CPU.
 */
class CpuRectifier final : public Rectifier
{
public:
  CpuRectifier();
  ~CpuRectifier() override = default;

private:
  common::Image process_impl(const common::Image & image) override;

  CameraInfo prepare_maps(const CameraInfo & camera_info) override;

  cv::Mat map_x_;
  cv::Mat map_y_;
};
}  // namespace accelerated_image_processor::pipeline
