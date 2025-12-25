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

#include "accelerated_image_processor_compression/jpeg.hpp"

#include <accelerated_image_processor_common/helper.hpp>

#include <cstring>
#include <stdexcept>

#if defined(JETSON_AVAILABLE) || defined(NVJPEG_AVAILABLE)
#include <nppi_color_conversion.h>
#include <nppi_data_exchange_and_initialization.h>
#endif

namespace accelerated_image_processor::compression
{
#ifdef JETSON_AVAILABLE
JetsonJPEGCompressor::JetsonJPEGCompressor() : JPEGCompressor()
{
  cudaStreamCreate(&stream_);
  encoder_ = NvJPEGEncoder::createJPEGEncoder("jpeg_encoder");
}

JetsonJPEGCompressor::~JetsonJPEGCompressor()
{
  if (image_d_) {
    nppiFree(image_d_);
    image_d_ = nullptr;
  }
  delete encoder_;
  cudaStreamDestroy(stream_);
}

common::Image JetsonJPEGCompressor::process_impl(const common::Image & image)
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
    static_cast<void *>(image_d_), image_step_bytes_, static_cast<const void *>(image.data.data()),
    image.step, image.step * sizeof(Npp8u), image.height, cudaMemcpyHostToDevice, stream_));

  NppiSize roi = {static_cast<int>(image.width), static_cast<int>(image.height)};
  if (image.format == common::ImageFormat::BGR) {
    constexpr int order[3] = {2, 1, 0};
    CHECK_NPP(nppiSwapChannels_8u_C3IR_Ctx(image_d_, image_step_bytes_, roi, order, context_));
  }

  CHECK_NPP(nppiRGBToYUV420_8u_C3P3R_Ctx(
    image_d_, image_step_bytes_, yuv_d_.data(), yuv_step_bytes_.data(), roi, context_));

  NvBuffer::NvBufferPlane & plane_y = buffer_->planes[0];
  NvBuffer::NvBufferPlane & plane_u = buffer_->planes[1];
  NvBuffer::NvBufferPlane & plane_v = buffer_->planes[2];
  CHECK_CUDA(cudaMemcpy2DAsync(
    plane_y.data, plane_y.fmt.stride, yuv_d_[0], yuv_step_bytes_[0], image.width, image.height,
    cudaMemcpyDeviceToHost, stream_));
  CHECK_CUDA(cudaMemcpy2DAsync(
    plane_u.data, plane_u.fmt.stride, yuv_d_[1], yuv_step_bytes_[1], image.width / 2,
    image.height / 2, cudaMemcpyDeviceToHost, stream_));
  CHECK_CUDA(cudaMemcpy2DAsync(
    plane_v.data, plane_v.fmt.stride, yuv_d_[2], yuv_step_bytes_[2], image.width / 2,
    image.height / 2, cudaMemcpyDeviceToHost, stream_));

  cudaStreamSynchronize(stream_);

  size_t out_buf_size = image.width * image.height * 3 / 2;
  unsigned char * out_data = new unsigned char[out_buf_size];

  // encodeFromBuffer only support YUV420
  CHECK_ERROR(
    encoder_->encodeFromBuffer(
      buffer_.value(), JCS_YCbCr, &out_data, out_buf_size, this->quality()),
    "NvJpeg Encoder Error");

  common::Image output;
  output.frame_id = image.frame_id;
  output.timestamp = image.timestamp;
  output.height = image.height;
  output.width = image.width;
  output.step = 0;  // 0 means this value is pointless because it's compressed
  output.format = image.format;
  output.data.resize(static_cast<size_t>(out_buf_size) / sizeof(uint8_t));
  std::memcpy(output.data.data(), out_data, out_buf_size);

  delete[] out_data;
  out_data = nullptr;

  return output;
}
#endif  // JETSON_AVAILABLE

#ifdef NVJPEG_AVAILABLE
NvJPEGCompressor::NvJPEGCompressor()
{
  CHECK_CUDA(cudaStreamCreate(&stream_));
  CHECK_NVJPEG(nvjpegCreateSimple(&handle_));
  CHECK_NVJPEG(nvjpegEncoderStateCreate(handle_, &state_, stream_));
  CHECK_NVJPEG(nvjpegEncoderParamsCreate(handle_, &params_, stream_));

  nvjpegEncoderParamsSetSamplingFactors(params_, NVJPEG_CSS_420, stream_);

  std::memset(&nv_image_, 0, sizeof(nv_image_));
}

NvJPEGCompressor::~NvJPEGCompressor()
{
  CHECK_NVJPEG(nvjpegEncoderParamsDestroy(params_));
  CHECK_NVJPEG(nvjpegEncoderStateDestroy(state_));
  CHECK_NVJPEG(nvjpegDestroy(handle_));
  CHECK_CUDA(cudaStreamDestroy(stream_));
}

common::Image NvJPEGCompressor::process_impl(const common::Image & image)
{
  nvjpegEncoderParamsSetQuality(params_, quality(), stream_);

  nvjpegInputFormat_t format;
  if (image.format == common::ImageFormat::RGB) {
    format = NVJPEG_INPUT_RGBI;
  } else if (image.format == common::ImageFormat::BGR) {
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
  output.format = image.format;
  output.data.resize(out_buffer_size);

  CHECK_NVJPEG(
    nvjpegEncodeRetrieveBitstream(handle_, state_, output.data.data(), &out_buffer_size, stream_));

  CHECK_CUDA(cudaStreamSynchronize(stream_));

  return output;
}

void NvJPEGCompressor::initialize_nv_image(const common::Image & image)
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
#endif  // NVJPEG_AVAILABLE

#ifdef TURBOJPEG_AVAILABLE
CpuJPEGCompressor::CpuJPEGCompressor() : JPEGCompressor()
{
  handle_ = tjInitCompress();
}

CpuJPEGCompressor::~CpuJPEGCompressor()
{
  if (buffer_) {
    tjFree(buffer_);
    buffer_ = nullptr;
  }
  tjDestroy(handle_);
}

common::Image CpuJPEGCompressor::process_impl(const common::Image & image)
{
  int tjpf;
  if (image.format == common::ImageFormat::RGB) {
    tjpf = TJPF_RGB;
  } else if (image.format == common::ImageFormat::BGR) {
    tjpf = TJPF_BGR;
  } else {
    throw std::runtime_error("Unsupported image format");
  }
  constexpr int sampling = TJ_420;

  int result = tjCompress2(
    handle_, image.data.data(), image.width, 0, image.height, tjpf, &buffer_, &size_, sampling,
    this->quality(), TJFLAG_FASTDCT);

#if defined(LIBJPEG_TURBO_VERSION) && (LIBJPEG_TURBO_VERSION >= 2)
  CHECK_ERROR(result != 0, tjGetErrorStr2(handle_));
#else
  CHECK_ERROR(result != 0, tjGetErrorStr());
#endif

  common::Image output;
  output.frame_id = image.frame_id;
  output.timestamp = image.timestamp;
  output.height = image.height;
  output.width = image.width;
  output.step = 0;  // 0 means this value is pointless because it's compressed
  output.format = image.format;
  output.data.resize(size_ / sizeof(uint8_t));
  std::memcpy(output.data.data(), buffer_, size_);

  return output;
}
#endif  // TURBOJPEG_AVAILABLE
}  // namespace accelerated_image_processor::compression
