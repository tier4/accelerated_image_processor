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

#include <array>
#include <optional>

#if defined(JETSON_AVAILABLE) || defined(NVJPEG_AVAILABLE)
#include <cuda_runtime.h>
#endif

#ifdef JETSON_AVAILABLE
#include <NvJpegEncoder.h>
#include <nppi_support_functions.h>
#endif

#ifdef NVJPEG_AVAILABLE
#include <nvjpeg.h>
#endif

#ifdef TURBOJPEG_AVAILABLE
#include <turbojpeg.h>
#endif

namespace accelerated_image_processor::compression
{
/**
 * @brief Abstract base class for JPEG compressors.
 */
class JPEGCompressor : public common::BaseProcessor
{
public:
  explicit JPEGCompressor(common::ParameterMap dedicated_parameters = {})
  : BaseProcessor(dedicated_parameters += {{"quality", 90}})
  {
  }

  ~JPEGCompressor() override = default;

  /**
   * @brief Return the quality of the JPEG compression.
   */
  int quality() const { return this->parameter_value<int>("quality"); }
};

#ifdef JETSON_AVAILABLE
/**
 * @brief JPEG compressor working on Jetson devices.
 */
class JetsonJPEGCompressor final : public JPEGCompressor
{
public:
  JetsonJPEGCompressor();
  ~JetsonJPEGCompressor() override;

private:
  common::Image process_impl(const common::Image & image) override;

  NvJPEGEncoder * encoder_;  //!< NvJPEG encoder handle.

  size_t image_size_{0};  //!< Size of the input image data.
  Npp8u * image_d_;       //!< Input image data in device memory.
  int image_step_bytes_;  //!< Step size in bytes for the input image data.

  std::array<Npp8u *, 3> yuv_d_;       //!< YUV data in device memory.
  std::array<int, 3> yuv_step_bytes_;  //!< Step sizes in bytes for the YUV data.

  cudaStream_t stream_;             //!< CUDA stream for asynchronous operations.
  NppStreamContext context_;        //!< NPP stream context for asynchronous operations.
  std::optional<NvBuffer> buffer_;  //!< Optional NvBuffer for storing encoded JPEG data.
};
#endif  // JETSON_AVAILABLE

#ifdef NVJPEG_AVAILABLE
/**
 * @brief JPEG compressor using NVJPEG library.
 */
class NvJPEGCompressor final : public JPEGCompressor
{
public:
  NvJPEGCompressor();
  ~NvJPEGCompressor() override;

private:
  common::Image process_impl(const common::Image & image) override;

  void initialize_nv_image(const common::Image & image);

  cudaStream_t stream_;                    //!< CUDA stream for asynchronous operations.
  nvjpegHandle_t handle_;                  //!< NVJPEG handle for JPEG encoding.
  nvjpegEncoderState_t state_;             //!< NVJPEG encoder state.
  nvjpegEncoderParams_t params_;           //!< NVJPEG encoder parameters.
  nvjpegInputFormat_t input_format_;       //!< NVJPEG input format.
  nvjpegChromaSubsampling_t subsampling_;  //!< NVJPEG chroma subsampling.
  nvjpegImage_t nv_image_;                 //!< NVJPEG image buffer.
};
#endif  // NVJPEG_AVAILABLE

#ifdef TURBOJPEG_AVAILABLE
/**
 * @brief JPEG compressor using CPU (TurboJPEG) library.
 */
class CpuJPEGCompressor final : public JPEGCompressor
{
public:
  CpuJPEGCompressor();
  ~CpuJPEGCompressor() override;

private:
  common::Image process_impl(const common::Image & image) override;

  tjhandle handle_ = nullptr;         //!< TurboJPEG handle for JPEG encoding.
  unsigned char * buffer_ = nullptr;  //!< Buffer for JPEG data.
  size_t size_ = 0;                   //!< Size of the buffer for JPEG data.
};
#endif  // TURBOJPEG_AVAILABLE
}  // namespace accelerated_image_processor::compression
