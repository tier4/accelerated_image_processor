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

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <memory>

namespace accelerated_image_processor::pipeline
{
/**
 * @brief Rectifier using OpenCV CPU.
 */
class CpuRectifier final : public Rectifier
{
public:
  CpuRectifier() : Rectifier() {}
  ~CpuRectifier() override = default;

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
    cv::Mat dst(image.height, image.width, CV_8UC3, reinterpret_cast<void *>(result.data.data()));

    cv::remap(src, dst, map_x_, map_y_, cv::INTER_LINEAR);

    return result;
  }

  CameraInfo prepare_maps(const CameraInfo & camera_info) override
  {
    map_x_ = cv::Mat(camera_info.height, camera_info.width, CV_32FC1);
    map_y_ = cv::Mat(camera_info.height, camera_info.width, CV_32FC1);

    return compute_maps(camera_info, map_x_.ptr<float>(), map_y_.ptr<float>(), alpha());
  }

  cv::Mat map_x_;
  cv::Mat map_y_;
};

std::unique_ptr<Rectifier> make_cpu_rectifier()
{
  return std::make_unique<CpuRectifier>();
}
}  // namespace accelerated_image_processor::pipeline
