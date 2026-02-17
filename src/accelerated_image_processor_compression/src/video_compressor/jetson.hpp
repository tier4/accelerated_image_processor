// Copyright 2026 TIER IV, Inc.
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
#include "accelerated_image_processor_common/helper.hpp"
#include "accelerated_image_processor_compression/video_compressor.hpp"
#include "jetson_error_helper.hpp"
#include "jetson_precise_timestamp_map.hpp"

#include <atomic>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef JETSON_AVAILABLE
#include "NvBufSurface.h"
#include "NvUtils.h"
#include "NvVideoEncoder.h"

#include <cuda_runtime.h>
#include <linux/videodev2.h>
#include <vpi/CUDAInterop.h>
#include <vpi/VPI.h>
#include <vpi/algo/ConvertImageFormat.h>
#endif

namespace accelerated_image_processor::compression
{
#ifdef JETSON_AVAILABLE
/**
 * @brief Enumeration of available codecs
 */
enum class SupportedCodec : uint8_t { H264, H265, AV1 };

/**
 * @brief Abstract base class for Video compressor working on Jetson devices.
 */
class JetsonVideoCompressor : public VideoCompressor
{
protected:
  /**
   * @brief Arguments passed to the capture plane dequeue callback.
   *
   * This structure holds information about the frame being processed,
   * including its identifier, dimensions, and a pointer to the owning
   * JetsonVideoCompressor instance. It is used to associate
   * the callback with the correct compressor context.
   *
   * @var std::string frame_id
   *   The frame ID derived from the input
   * @var int input_width
   *   Width of the input image in pixels.
   * @var int input_height
   *   Height of the input image in pixels.
   * @var JetsonVideoCompressor* obj
   *   Pointer to the compressor instance that owns this callback.
   */
  struct DqCallbackArgs
  {
    std::string frame_id;
    int input_width;
    int input_height;
    JetsonVideoCompressor * obj;

    DqCallbackArgs(
      const std::string f_id, const int width, const int height, JetsonVideoCompressor * obj)
    : frame_id(f_id), input_width(width), input_height(height), obj(obj)
    {
    }
  };

public:
  /**
   * @brief Lookup table to tie the string to the corresponding codecs
   */
  inline static const std::unordered_map<std::string, SupportedCodec> supported_codec_map = {
    {"H264", SupportedCodec::H264},
    {"H265", SupportedCodec::H265},
    {"AV1", SupportedCodec::AV1},
  };

  /**
   * @brief Lookup table to tie the supported codecs enumeration to the common::ImageFormat
   */
  inline static const std::unordered_map<SupportedCodec, common::ImageFormat>
    supported_codec_format_map = {
      {SupportedCodec::H264, common::ImageFormat::H264},
      {SupportedCodec::H265, common::ImageFormat::H265},
      {SupportedCodec::AV1, common::ImageFormat::AV1},
    };

  /**
   * @brief Map between strings and corresponding hardware preset types
   */
  inline static const std::unordered_map<std::string, v4l2_enc_hw_preset_type> hardware_preset_map =
    {
      {"DISABLE", V4L2_ENC_HW_PRESET_DISABLE},     {"SLOW", V4L2_ENC_HW_PRESET_SLOW},
      {"MEDIUM", V4L2_ENC_HW_PRESET_MEDIUM},       {"FAST", V4L2_ENC_HW_PRESET_FAST},
      {"ULTRAFAST", V4L2_ENC_HW_PRESET_ULTRAFAST},
    };

  /**
   * @brief Map between compression type and pixel format to be used for configuring encoder input
   * (consumed by encoder API)
   */
  inline static const std::unordered_map<VideoCompressionType, __u32> pixel_format_map = {
    {VideoCompressionType::LOSSY, V4L2_PIX_FMT_NV24M},
    {VideoCompressionType::LOSSLESS, V4L2_PIX_FMT_NV12M},
  };

  /**
   * @brief Map between compression type and NvBuffer pixel format to be used for configuring
   * encoder input DMA buffer (consumed by NvBuffer API)
   */
  inline static const std::unordered_map<VideoCompressionType, NvBufSurfaceColorFormat>
    nvbuf_color_format_map = {
      {VideoCompressionType::LOSSY,
       NVBUF_COLOR_FORMAT_NV12_ER},  // Y/CbCr 4:2:0 multi-planar, extended range (full color)
      {VideoCompressionType::LOSSLESS,
       NVBUF_COLOR_FORMAT_NV24_ER},  // Y/CbCr 4:4:4 multi-planar, extended range (full color)
    };

  /**
   * @brief Configuration parameters for the Jetson video encoder.
   *
   * This structure encapsulates all the tunable settings that control the
   * behavior of the NvVideoEncoder.  The values are typically derived from
   * the user supplied parameter map and are validated during the
   * initialization phase.
   *
   * @var int buffer_length
   *   Number of buffers reserved for the encoder output plane.
   *
   * @var VideoCompressionType compression_type
   *   The compression mode (lossy or lossless) selected for the stream.
   *
   * @var int idr_interval
   *   Interval (in frames) between IDR (Instantaneous Decoder Refresh)
   *   keyframes.
   *
   * @var int i_frame_interval
   *   Interval (in frames) between I‑frames.
   *
   * @var int frame_rate_numerator
   *   Numerator of the target frame rate (numerator in second).
   *
   * @var int frame_rate_denominator
   *   Denominator of the target frame rate (denominator in frames).
   *
   * @var v4l2_enc_hw_preset_type hw_preset_type
   *   Hardware preset that tunes the encoder for speed or quality.
   *
   * @var bool use_max_performance_mode
   *   When true the encoder is forced into a high‑performance mode,
   *   potentially at the cost of increased power consumption.
   *
   * @var double target_bits_per_pixel
   *   Target bitrate expressed as bits per pixel.  This value is used
   *   by the encoder to determine the target bit rate, which mainly affects encoded image quality
   *   and payload size.
   */
  struct EncoderParameter
  {
    int buffer_length;
    VideoCompressionType compression_type;
    int idr_interval;
    int i_frame_interval;
    int frame_rate_numerator;
    int frame_rate_denominator;
    v4l2_enc_hw_preset_type hw_preset_type;
    bool use_max_performance_mode;
    double target_bits_per_pixel;
  };

  /**
   * @brief Constructor
   */
  explicit JetsonVideoCompressor(
    SupportedCodec codec, common::ParameterMap dedicated_parameters = {})
  : VideoCompressor(
      CompressorBackend::JETSON, dedicated_parameters +=
                                 {{"buffer_length", static_cast<int>(4)},
                                  {"use_max_performance", static_cast<bool>(true)},
                                  {"hw_preset_type", static_cast<std::string>("disable")},
                                  {"target_bits_per_pixel", static_cast<double>(0.1)}}),
    codec_(codec)
  {
    // To make EGL
    // (https://developer.nvidia.com/blog/egl-eye-opengl-visualization-without-x-server/) , which is
    // utilized VPI underhood, work headless, unset DISPLAY environment variable to avoid it affects
    // EGL behavior
    int replace_env = 1;
    setenv("EGL_PLATFORM", "surfaceless", replace_env);

    CHECK_CUDA(cudaStreamCreate(&stream_));
    uint64_t vpi_stream_flag = 0;  // No flag is specified
    CHECK_VPI(vpiStreamCreateWrapperCUDA(
      stream_, vpi_stream_flag, &vpi_stream_));  // Share stream between CUDA and VPI

    encoder_ = NvVideoEncoder::createVideoEncoder("encoder");
    if (!encoder_) {
      throw std::runtime_error("Failed to create NvVideoEncoder");
    }
  }

  /**
   * @brief Destructor: clean up the resources accordingly
   */
  ~JetsonVideoCompressor()
  {
    // Wait till capture plane DQ Thread finishes
    // i.e. all the capture plane buffers are dequeued
    encoder_->capture_plane.waitForDQThread(1000);

    if (input_rgb_dev_) {
      vpiImageDestroy(input_rgb_dev_);
    }

    if (output_yuv_dev_) {
      vpiImageDestroy(output_yuv_dev_);
    }

    for (uint32_t i = 0; i < encoder_->output_plane.getNumBuffers(); i++) {
      /* Unmap output plane buffer for memory type DMABUF. */
      CHECK_ERROR(
        encoder_->output_plane.unmapOutputBuffers(i, output_plane_fds_[i]) < 0,
        "Error while unmapping buffer at output plane");

      // ERROR_CHECK(NvBufSurf::NvDestroy(output_plane_fds_[i]), "Failed to Destroy NvBuffer");
      NvBufSurfaceDestroy(output_nvsurface_[i]);
      output_plane_fds_[i] = -1;
    }

    vpiStreamDestroy(vpi_stream_);  // this returns void, so VPI_CHECK cannot be applicable
    CHECK_CUDA(cudaStreamDestroy(stream_));
  }

  /**
   * @brief [override] process the input image without postprocessing
   */
  std::optional<common::Image> process(const common::Image & image) override
  {
    if (!is_ready()) {
      std::cerr << "JetsonVideoCompressor is not ready. Skip this frame" << std::endl;
      return std::nullopt;
    }

    auto processed = this->process_impl(image);
    // always return nullopt because processed (encoded) result will be handled in the other thread
    return std::nullopt;
  }

  /**
   * @brief [override] Check the encoder is ready to run processing.
   */
  bool is_ready() const override { return state_ != State::ERROR; }

  // Getter functions to access members from static `encoder_capture_plane_dq_callback`
  auto * encoder() { return this->encoder_; }
  const auto & encoder_params() { return this->encoder_params_; }
  auto & initial_frame_count() { return this->initial_frame_count_; }
  auto & timestamp_map() { return this->timestamp_map_; }
  const auto & codec() { return this->codec_; }

protected:
  enum class State : uint8_t { UNINITIALIZED, READY, ERROR };
  struct PayloadInfo
  {
    uint8_t * payload_ptr;
    size_t payload_size;
    size_t offset;
  };
  State state_{State::UNINITIALIZED};

  /**
   * @brief codec dedicated parameter collection
   */
  virtual EncResult collect_codec_params_impl() = 0;

  /**
   * @brief codec dedicated setup steps for capture plane (encoder output)
   */
  virtual EncResult set_capture_plane_format_impl(
    const uint32_t & width, const uint32_t & height, const uint32_t & image_size) = 0;

  /**
   * @brief codec dedicated initialization steps
   */
  virtual EncResult init_codec_impl() = 0;

  /**
   * @brief Payload preprocessing implementation
   * Some codecs may need dedicated handling for the encoded payload, which this function handles
   */
  virtual PayloadInfo payload_preprocess_impl(
    [[maybe_unused]] DqCallbackArgs * callback_args, const size_t bytes_used,
    const NvBuffer * buffer)
  {
    uint8_t * payload_ptr = reinterpret_cast<uint8_t *>(buffer->planes[0].data);
    size_t payload_size = bytes_used;
    size_t offset = 0;

    return {payload_ptr, payload_size, offset};
  }

  /**
   * @brief Codec dedicated payload copy implementation
   * Similar to payload preprocess, this function handles codec dedicated data copy
   */
  virtual void payload_copy_impl(
    [[maybe_unused]] const bool is_keyframe, const PayloadInfo & payload_info,
    [[maybe_unused]] DqCallbackArgs * callback_args, std::vector<uint8_t> & copy_destination)
  {
    auto & [payload_ptr, payload_size, offset] = payload_info;
    copy_destination.resize(payload_size);
    std::memcpy(copy_destination.data(), payload_ptr, payload_size);
  }

  NvVideoEncoder * encoder_;
  EncoderParameter encoder_params_;

private:
  EncResult collect_params(EncoderParameter & params);
  EncResult init_encoder(const common::Image & image);
  EncResult setup_output_plane(const int & height, const int & width);
  void fill_encoder_input_async(void);
  common::Image process_impl(const common::Image & image) override;
  inline EncStatus record_error(const std::string & msg)
  {
    last_error_ = msg;
    state_ = State::ERROR;
    return EncStatus(false, msg);
  }

  static bool encoder_capture_plane_dq_callback(
    struct v4l2_buffer * v4l2_buffer, NvBuffer * buffer, [[maybe_unused]] NvBuffer * shared_buffer,
    void * arg);

  std::string last_error_{""};
  SupportedCodec codec_;

  std::vector<int> output_plane_fds_{};
  std::vector<NvBufSurface *> output_nvsurface_{};
  VPIImage input_rgb_dev_{nullptr};
  VPIImage output_yuv_dev_{nullptr};
  cudaStream_t stream_{};
  VPIStream vpi_stream_{};

  std::unique_ptr<TimestampMap> timestamp_map_;
  std::unique_ptr<DqCallbackArgs> dq_callback_args_;

  int initial_frame_count_{0};
};
#endif  // JETSON_AVAILABLE
}  // namespace accelerated_image_processor::compression
