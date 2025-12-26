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

#include "accelerated_image_processor_pipeline/rectifier.hpp"
#include "utility.hpp"

#ifdef OPENCV_CUDA_AVAILABLE
#include <opencv2/core/cuda.hpp>
#endif

namespace accelerated_image_processor::pipeline
{
#ifdef OPENCV_CUDA_AVAILABLE
/**
 * @brief Rectifier using OpenCV CUDA.
 */
class OpenCvCudaRectifier final : public Rectifier
{
public:
  OpenCvCudaRectifier() : Rectifier(RectifierBackend::OPENCV_CUDA) {}
  ~OpenCvCudaRectifier() override = default;

private:
  common::Image process_impl(const common::Image & image) override
  {
    common::Image result;
    result.height = image.height;
    result.width = image.width;
    result.format = image.format;
    result.step = image.step;
    result.data.resize(image.data.size());

    cv::Mat src(image.height, image.width, CV_8UC3, const_cast<unsigned char *>(image.data.data()));
    cv::cuda::GpuMat d_src = cv::cuda::GpuMat(src);
    cv::cuda::GpuMat d_dst = cv::cuda::GpuMat(cv::Size(image.width, image.height), src.type());

    cv::cuda::remap(d_src, d_dst, map_x_, map_y_, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

    // copy back to result
    cv::Mat dst(image.height, image.width, CV_8UC3, reinterpret_cast<void *>(result.data.data()));
    d_dst.download(dst);

    return result;
  }

  CameraInfo prepare_maps(const CameraInfo & camera_info) override
  {
    cv::Mat map_x(camera_info.height, camera_info.width, CV_32FC1);
    cv::Mat map_y(camera_info.height, camera_info.width, CV_32FC1);

    CameraInfo camera_info_rect =
      compute_maps_opencv(camera_info, map_x.ptr<float>(), map_y.ptr<float>(), this->alpha());

    map_x_ = cv::cuda::GpuMat(map_x);
    map_y_ = cv::cuda::GpuMat(map_y);

    return camera_info_rect;
  }

  cv::cuda::GpuMat map_x_;
  cv::cuda::GpuMat map_y_;
};

std::unique_ptr<Rectifier> make_opencv_cuda_rectifier()
{
  return std::make_unique<OpenCvCudaRectifier>();
}
#else
std::unique_ptr<Rectifier> make_opencv_cuda_rectifier()
{
  return nullptr;
}
#endif
}  // namespace accelerated_image_processor::pipeline
