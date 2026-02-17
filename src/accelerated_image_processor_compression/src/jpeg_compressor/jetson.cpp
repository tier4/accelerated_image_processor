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
#include <optional>
#include <string>

#ifdef JETSON_AVAILABLE
#include <NvJpegEncoder.h>
#include <cuda_runtime.h>
#include <nppi_color_conversion.h>
#include <nppi_data_exchange_and_initialization.h>
#include <nppi_support_functions.h>
#endif  // JETSON_AVAILABLE

namespace accelerated_image_processor::compression
{
#ifdef JETSON_AVAILABLE
/**
 * @brief JPEG compressor working on Jetson devices.
 */
class JetsonJPEGCompressor final : public JPEGCompressor
{
public:
  JetsonJPEGCompressor() : JPEGCompressor(CompressorBackend::JETSON)
  {
    CHECK_CUDA(cudaStreamCreate(&stream_));
    encoder_ = NvJPEGEncoder::createJPEGEncoder("jpeg_encoder");
  }
  ~JetsonJPEGCompressor() override
  {
    if (image_d_) {
      nppiFree(image_d_);
      image_d_ = nullptr;
    }
    for (auto & p : yuv_d_) {
      if (p) {
        nppiFree(p);
        p = nullptr;
      }
    }
    if (encoder_) {
      delete encoder_;
      encoder_ = nullptr;
    }
    if (stream_) {
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
  }

private:
  common::Image process_impl(const common::Image & image) override
  {
    if (image_size_ < image.data.size()) {
      image_d_ = nppiMalloc_8u_C3(image.width, image.height, &image_step_bytes_);
      image_size_ = image.data.size();

      yuv_d_[0] = nppiMalloc_8u_C1(image.width, image.height, &yuv_step_bytes_[0]);          // Y
      yuv_d_[1] = nppiMalloc_8u_C1(image.width / 2, image.height / 2, &yuv_step_bytes_[1]);  // U
      yuv_d_[2] = nppiMalloc_8u_C1(image.width / 2, image.height / 2, &yuv_step_bytes_[2]);  // V

      // fill elements of nppStreamContext
      {
        context_.hStream = stream_;
        cudaGetDevice(&context_.nCudaDeviceId);
        cudaDeviceProp dev_prop;
        cudaGetDeviceProperties(&dev_prop, context_.nCudaDeviceId);
        context_.nMultiProcessorCount = dev_prop.multiProcessorCount;
        context_.nMaxThreadsPerMultiProcessor = dev_prop.maxThreadsPerMultiProcessor;
        context_.nMaxThreadsPerBlock = dev_prop.maxThreadsPerBlock;
        context_.nSharedMemPerBlock = dev_prop.sharedMemPerBlock;
        cudaDeviceGetAttribute(
          &context_.nCudaDevAttrComputeCapabilityMajor, cudaDevAttrComputeCapabilityMajor,
          context_.nCudaDeviceId);
        cudaDeviceGetAttribute(
          &context_.nCudaDevAttrComputeCapabilityMinor, cudaDevAttrComputeCapabilityMinor,
          context_.nCudaDeviceId);
        cudaStreamGetFlags(context_.hStream, &context_.nStreamFlags);
      }

      buffer_.emplace(V4L2_PIX_FMT_YUV420M, image.width, image.height, 0);
      CHECK_ERROR(buffer_->allocateMemory() != 0, "NvBuffer allocation failed");

      encoder_->setCropRect(0, 0, image.width, image.height);
    }

    CHECK_CUDA(cudaMemcpy2DAsync(
      static_cast<void *>(image_d_), image_step_bytes_,
      static_cast<const void *>(image.data.data()), image.step, image.step * sizeof(Npp8u),
      image.height, cudaMemcpyHostToDevice, stream_));

    NppiSize roi = {static_cast<int>(image.width), static_cast<int>(image.height)};
    if (image.encoding == common::ImageEncoding::BGR) {
      constexpr int order[3] = {2, 1, 0};
      CHECK_NPP(nppiSwapChannels_8u_C3IR_Ctx(image_d_, image_step_bytes_, roi, order, context_));
    }

    CHECK_NPP(nppiRGBToYUV420_8u_C3P3R_Ctx(
      image_d_, image_step_bytes_, yuv_d_.data(), yuv_step_bytes_.data(), roi, context_));

    NvBuffer::NvBufferPlane & plane_y = buffer_->planes[0];
    NvBuffer::NvBufferPlane & plane_u = buffer_->planes[1];
    NvBuffer::NvBufferPlane & plane_v = buffer_->planes[2];

    // Copy YUV planes into the NvBuffer CPU-mapped pointers.
    CHECK_CUDA(cudaMemcpy2DAsync(
      plane_y.data, plane_y.fmt.stride, yuv_d_[0], yuv_step_bytes_[0], image.width, image.height,
      cudaMemcpyDeviceToHost, stream_));
    CHECK_CUDA(cudaMemcpy2DAsync(
      plane_u.data, plane_u.fmt.stride, yuv_d_[1], yuv_step_bytes_[1], image.width / 2,
      image.height / 2, cudaMemcpyDeviceToHost, stream_));
    CHECK_CUDA(cudaMemcpy2DAsync(
      plane_v.data, plane_v.fmt.stride, yuv_d_[2], yuv_step_bytes_[2], image.width / 2,
      image.height / 2, cudaMemcpyDeviceToHost, stream_));

    // Ensure all copies are complete before touching NvBuffer metadata / encoding.
    CHECK_CUDA(cudaStreamSynchronize(stream_));

    size_t out_buf_size = image.width * image.height * 3 / 2;
    unsigned char * out_data = new unsigned char[out_buf_size];

    CHECK_ERROR(
      encoder_->encodeFromBuffer(buffer_.value(), JCS_YCbCr, &out_data, out_buf_size, quality()),
      "NvJPEGEncoder::encodeFromBuffer failed (non-zero return code)");

    common::Image output;
    output.frame_id = image.frame_id;
    output.timestamp = image.timestamp;
    output.height = image.height;
    output.width = image.width;
    output.step = 0;  // 0 means this value is pointless because it's compressed
    output.encoding = image.encoding;
    output.format = common::ImageFormat::JPEG;
    output.data.resize(static_cast<size_t>(out_buf_size) / sizeof(uint8_t));
    memcpy(output.data.data(), out_data, out_buf_size);

    delete[] out_data;
    out_data = nullptr;

    return output;
  }

  NvJPEGEncoder * encoder_;  //!< NvJPEG encoder handle.

  size_t image_size_{0};      //!< Size of the input image data.
  Npp8u * image_d_{nullptr};  //!< Input image data in device memory.
  int image_step_bytes_{0};   //!< Step size in bytes for the input image data.

  std::array<Npp8u *, 3> yuv_d_{{nullptr, nullptr, nullptr}};  //!< YUV data in device memory.
  std::array<int, 3> yuv_step_bytes_{{0, 0, 0}};  //!< Step sizes in bytes for the YUV data.

  cudaStream_t stream_{nullptr};    //!< CUDA stream for asynchronous operations.
  NppStreamContext context_{};      //!< NPP stream context for asynchronous operations.
  std::optional<NvBuffer> buffer_;  //!< Optional NvBuffer for storing encoded JPEG data.
};

std::unique_ptr<JPEGCompressor> make_jetsonjpeg_compressor()
{
  return std::make_unique<JetsonJPEGCompressor>();
}
#else
std::unique_ptr<JPEGCompressor> make_jetsonjpeg_compressor()
{
  return nullptr;
}
#endif  // JETSON_AVAILABLE
}  // namespace accelerated_image_processor::compression
