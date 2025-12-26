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

#include <accelerated_image_processor_common/helper.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <memory>

namespace accelerated_image_processor::pipeline
{
namespace
{
static CameraInfo compute_maps(const CameraInfo & info, float * map_x, float * map_y, double alpha)
{
  cv::Mat intrinsics(3, 3, CV_64F);
  cv::Mat distortion_coefficients(1, info.d.size(), CV_64F);

  for (size_t row = 0; row < 3; row++) {
    for (size_t col = 0; col < 3; col++) {
      intrinsics.at<double>(row, col) = info.k[row * 3 + col];
    }
  }

  for (std::size_t col = 0; col < info.d.size(); col++) {
    distortion_coefficients.at<double>(col) = info.d[col];
  }

  cv::Mat new_intrinsics = cv::getOptimalNewCameraMatrix(
    intrinsics, distortion_coefficients, cv::Size(info.width, info.height), alpha);

  cv::Mat m1(info.height, info.width, CV_32FC1, map_x);
  cv::Mat m2(info.height, info.width, CV_32FC1, map_y);

  cv::initUndistortRectifyMap(
    intrinsics, distortion_coefficients, cv::Mat::eye(3, 3, CV_64F), new_intrinsics,
    cv::Size(info.width, info.height), CV_32FC1, m1, m2);

  // Copy the original camera info and update only D and K
  CameraInfo camera_info_rect(info);
  // After undistortion, the result will be as if it is captured with a camera using
  // the camera with new_intrinsics and zero distortion
  camera_info_rect.d.assign(info.d.size(), 0.);
  for (size_t row = 0; row < 3; row++) {
    for (size_t col = 0; col < 3; col++) {
      camera_info_rect.k[row * 3 + col] = new_intrinsics.at<double>(row, col);
      camera_info_rect.p[row * 4 + col] = new_intrinsics.at<double>(row, col);
    }
  }
  return camera_info_rect;
}
}  // namespace

#ifdef NPP_AVAILABLE
NPPRectifier::NPPRectifier() : Rectifier()
{
  cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
  nppSetStream(stream_);
}

NPPRectifier::~NPPRectifier()
{
  if (map_x_ != nullptr) {
    nppiFree(map_x_);
    map_x_ = nullptr;
  }
  if (map_y_ != nullptr) {
    nppiFree(map_y_);
    map_y_ = nullptr;
  }
  if (src_ != nullptr) {
    nppiFree(src_);
    src_ = nullptr;
  }
  if (dst_ != nullptr) {
    nppiFree(dst_);
    dst_ = nullptr;
  }
  cudaStreamDestroy(stream_);
}

common::Image NPPRectifier::process_impl(const common::Image & image)
{
  common::Image result;
  result.height = image.height;
  result.width = image.width;
  result.format = image.format;
  result.step = image.step;
  result.data.resize(image.data.size());

  NppiRect src_roi = {0, 0, static_cast<int>(image.width), static_cast<int>(image.height)};
  NppiSize src_size = {static_cast<int>(image.width), static_cast<int>(image.height)};
  NppiSize dst_roi_size = {static_cast<int>(image.width), static_cast<int>(image.height)};

  CHECK_CUDA(cudaMemcpy2DAsync(
    src_, src_step_, image.data.data(), image.step, image.width * 3, image.height,
    cudaMemcpyHostToDevice, stream_));

  CHECK_NPP(nppiRemap_8u_C3R(
    src_, src_size, src_step_, src_roi, map_x_, map_x_step_, map_y_, map_y_step_, dst_, dst_step_,
    dst_roi_size, NPPI_INTER_LINEAR));

  CHECK_CUDA(cudaMemcpy2DAsync(
    static_cast<void *>(result->data.data()), result->step, static_cast<const void *>(dst_),
    dst_step_,
    image.width * 3 * sizeof(Npp8u),  // in byte
    image.height, cudaMemcpyDeviceToHost, stream_));

  return result;
}

CameraInfo NPPRectifier::prepare_maps(const CameraInfo & camera_info)
{
  map_x_ = nppiMalloc_32f_C1(camera_info.width, camera_info.height, &map_x_step_);
  map_y_ = nppiMalloc_32f_C1(camera_info.width, camera_info.height, &map_y_step_);

  src_ = nppiMalloc_8u_C3(camera_info.width, camera_info.height, &src_step_);
  dst_ = nppiMalloc_8u_C3(camera_info.width, camera_info.height, &dst_step_);

  float * map_x = new float[camera_info.width * camera_info.height];
  float * map_y = new float[camera_info.width * camera_info.height];

  camera_info_rect = compute_maps_opencv(camera_info, map_x, map_y, alpha);

  delete[] map_x;
  delete[] map_y;

  return camera_info_rect;
}
#endif  // NPP_AVAILABLE

#ifdef OPENCV_CUDA_AVAILABLE
OpenCVCUDARectifier::OpenCVCUDARectifier() : Rectifier()
{
}

CameraInfo OpenCVCUDARectifier::prepare_maps(const CameraInfo & camera_info)
{
  cv::Mat map_x(camera_info.height, camera_info.width, CV_32FC1);
  cv::Mat map_y(camera_info.height, camera_info.width, CV_32FC1);

  camera_info_rect =
    compute_maps_opencv(camera_info, map_x.ptr<float>(), map_y.ptr<float>(), alpha);

  map_x_ = cv::cuda::GpuMat(map_x);
  map_y_ = cv::cuda::GpuMat(map_y);

  return camera_info_rect;
}

void OpenCVCUDARectifier::process_impl(const common::Image & image)
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

#endif  // OPENCV_CUDA_AVAILABLE

CpuRectifier::CpuRectifier() : Rectifier()
{
}

common::Image CpuRectifier::process_impl(const common::Image & image)
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

CameraInfo CpuRectifier::prepare_maps(const CameraInfo & camera_info)
{
  map_x_ = cv::Mat(camera_info.height, camera_info.width, CV_32FC1);
  map_y_ = cv::Mat(camera_info.height, camera_info.width, CV_32FC1);

  return compute_maps(camera_info, map_x_.ptr<float>(), map_y_.ptr<float>(), this->alpha());
}
}  // namespace accelerated_image_processor::pipeline
