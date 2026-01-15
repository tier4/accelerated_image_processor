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

#include "accelerated_image_processor_compression/jpeg_compressor.hpp"

#include <accelerated_image_processor_common/datatype.hpp>
#include <accelerated_image_processor_common/helper.hpp>

#include <cstring>
#include <memory>
#include <stdexcept>

#ifdef NVJPEG_AVAILABLE
#include <cuda_runtime.h>
#include <nppi_color_conversion.h>
#include <nppi_data_exchange_and_initialization.h>
#include <nvjpeg.h>
#endif  // NVJPEG_AVAILABLE

namespace accelerated_image_processor::compression
{
#ifdef NVJPEG_AVAILABLE
/**
 * @brief JPEG compressor using NVJPEG library.
 */
class NvJPEGCompressor final : public JPEGCompressor
{
public:
  NvJPEGCompressor() : JPEGCompressor(JPEGBackend::NVJPEG)
  {
    CHECK_CUDA(cudaStreamCreate(&stream_));
    CHECK_NVJPEG(nvjpegCreateSimple(&handle_));
    CHECK_NVJPEG(nvjpegEncoderStateCreate(handle_, &state_, stream_));
    CHECK_NVJPEG(nvjpegEncoderParamsCreate(handle_, &params_, stream_));

    nvjpegEncoderParamsSetSamplingFactors(params_, NVJPEG_CSS_420, stream_);

    std::memset(&nv_image_, 0, sizeof(nv_image_));
  }
  ~NvJPEGCompressor() override
  {
    CHECK_NVJPEG(nvjpegEncoderParamsDestroy(params_));
    CHECK_NVJPEG(nvjpegEncoderStateDestroy(state_));
    CHECK_NVJPEG(nvjpegDestroy(handle_));
    CHECK_CUDA(cudaStreamDestroy(stream_));
  }

private:
  common::Image process_impl(const common::Image & image) override
  {
    nvjpegEncoderParamsSetQuality(params_, quality(), stream_);

    nvjpegInputFormat_t format;
    if (image.encoding == common::ImageEncoding::RGB) {
      format = NVJPEG_INPUT_RGBI;
    } else if (image.encoding == common::ImageEncoding::BGR) {
      format = NVJPEG_INPUT_BGRI;
    } else {
      throw std::runtime_error("Unsupported image format");
    }
    initialize_nv_image(image);
    CHECK_NVJPEG(nvjpegEncodeImage(
      handle_, state_, params_, &nv_image_, format, image.width, image.height, stream_));

    size_t out_buffer_size = 0;
    CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(handle_, state_, NULL, &out_buffer_size, stream_));

    common::Image output;
    output.frame_id = image.frame_id;
    output.timestamp = image.timestamp;
    output.height = image.height;
    output.width = image.width;
    output.step = 0;  // 0 means this value is pointless because it's compressed
    output.encoding = image.encoding;
    output.format = common::ImageFormat::JPEG;
    output.data.resize(out_buffer_size);

    CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(
      handle_, state_, output.data.data(), &out_buffer_size, stream_));

    CHECK_CUDA(cudaStreamSynchronize(stream_));

    return output;
  }

  void initialize_nv_image(const common::Image & image)
  {
    if (nv_image_.channel[0] == nullptr) {
      CHECK_CUDA(cudaMallocAsync(
        reinterpret_cast<void **>(&nv_image_.channel[0]), image.data.size(), stream_));
    }
    CHECK_CUDA(cudaMemsetAsync(nv_image_.channel[0], 0, image.data.size(), stream_));
    CHECK_CUDA(cudaMemcpyAsync(
      nv_image_.channel[0], image.data.data(), image.data.size(), cudaMemcpyHostToDevice, stream_));

    constexpr int channels = 3;
    // NOTE: assume image is RGBI/BGRI
    nv_image_.pitch[0] = image.width * channels;
  }

  cudaStream_t stream_;           //!< CUDA stream for asynchronous operations.
  nvjpegHandle_t handle_;         //!< NVJPEG handle for JPEG encoding.
  nvjpegEncoderState_t state_;    //!< NVJPEG encoder state.
  nvjpegEncoderParams_t params_;  //!< NVJPEG encoder parameters.
  nvjpegImage_t nv_image_;        //!< NVJPEG image buffer.
};

std::unique_ptr<JPEGCompressor> make_nvjpeg_compressor()
{
  return std::make_unique<NvJPEGCompressor>();
}
#else
std::unique_ptr<JPEGCompressor> make_nvjpeg_compressor()
{
  return nullptr;
}
#endif  // NVJPEG_AVAILABLE
}  // namespace accelerated_image_processor::compression
