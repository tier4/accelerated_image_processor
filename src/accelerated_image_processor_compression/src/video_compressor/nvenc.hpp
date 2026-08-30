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
#include "nvenc_error_helper.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#ifdef NVENC_AVAILABLE
#include <cuda.h>
#include <cuda_runtime.h>
#include <ffnvcodec/nvEncodeAPI.h>
#include <nppi_color_conversion.h>
#endif

namespace accelerated_image_processor::compression
{
#ifdef NVENC_AVAILABLE
/**
 * @brief Enumeration of the codecs this backend supports
 *
 * NVENC hardware itself is capable of H264/H265 as well, but only the codecs listed here are
 * wired up by this package for now
 */
enum class NvencSupportedCodec : uint8_t { AV1 };

/**
 * @brief Abstract base class for video compressors working on discrete NVIDIA GPUs.
 *
 * This class drives the NVENC hardware encoder through NvEncodeAPI, whose headers are provided
 * by [nv-codec-headers](https://github.com/FFmpeg/nv-codec-headers). The counterpart of
 * JetsonVideoCompressor: while Jetson devices expose their encoder via the V4L2 based Jetson
 * Multimedia API, x86 hosts with a discrete GPU expose it via NvEncodeAPI
 * (`libnvidia-encode.so.1`, shipped with the NVIDIA display driver).
 *
 * The parameter keys are intentionally kept identical to the Jetson ones wherever the meaning
 * carries over, so that a single parameter file can drive both backends.
 *
 * Unlike JetsonVideoCompressor, which receives encoded frames on a dedicated dequeue thread,
 * this class encodes synchronously: process() feeds one frame and returns the encoded packet
 * for it (the registered postprocess is still invoked, hence a user code can be shared between
 * the both backends).
 */
class NvencVideoCompressor : public VideoCompressor
{
public:
  /**
   * @brief Lookup table to tie the string to the corresponding codecs
   */
  inline static const std::unordered_map<std::string, NvencSupportedCodec> supported_codec_map = {
    {"AV1", NvencSupportedCodec::AV1},
  };

  /**
   * @brief Lookup table to tie the supported codecs enumeration to the common::ImageFormat
   */
  inline static const std::unordered_map<NvencSupportedCodec, common::ImageFormat>
    supported_codec_format_map = {
      {NvencSupportedCodec::AV1, common::ImageFormat::AV1},
    };

  /**
   * @brief Map between strings and corresponding NVENC encode preset
   *
   * NVENC grades its presets from P1 (fastest) to P7 (slowest, best quality). To keep the
   * parameter file compatible with the Jetson backend, the Jetson hardware preset names are
   * accepted as aliases of the closest NVENC preset. `DISABLE`, which means "let the driver
   * pick" on Jetson, is mapped to the NVENC default grade (P4)
   */
  inline static const std::unordered_map<std::string, GUID> hardware_preset_map = {
    {"ULTRAFAST", NV_ENC_PRESET_P1_GUID}, {"FAST", NV_ENC_PRESET_P3_GUID},
    {"MEDIUM", NV_ENC_PRESET_P4_GUID},    {"SLOW", NV_ENC_PRESET_P6_GUID},
    {"DISABLE", NV_ENC_PRESET_P4_GUID},   {"P1", NV_ENC_PRESET_P1_GUID},
    {"P2", NV_ENC_PRESET_P2_GUID},        {"P3", NV_ENC_PRESET_P3_GUID},
    {"P4", NV_ENC_PRESET_P4_GUID},        {"P5", NV_ENC_PRESET_P5_GUID},
    {"P6", NV_ENC_PRESET_P6_GUID},        {"P7", NV_ENC_PRESET_P7_GUID},
  };

  /**
   * @brief Map between strings and the corresponding NVENC tuning information
   *
   * The tuning information tells the encoder which aspect the chosen preset should be biased
   * to. Note that `LOSSLESS` is not listed here because lossless encoding is requested via the
   * `compression_type` parameter, which is shared with the Jetson backend
   */
  inline static const std::unordered_map<std::string, NV_ENC_TUNING_INFO> tuning_info_map = {
    {"HIGH_QUALITY", NV_ENC_TUNING_INFO_HIGH_QUALITY},
    {"ULTRA_HIGH_QUALITY", NV_ENC_TUNING_INFO_ULTRA_HIGH_QUALITY},
    {"LOW_LATENCY", NV_ENC_TUNING_INFO_LOW_LATENCY},
    {"ULTRA_LOW_LATENCY", NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY},
  };

  /**
   * @brief Map between compression type and the NVENC input buffer format
   *
   * Only lossy encoding is listed: NVENC exposes lossless encoding for a limited set of codecs
   * only, and the availability is validated against the hardware capability in
   * validate_compression_type_compatibility() before this map is consulted
   */
  inline static const std::unordered_map<VideoCompressionType, NV_ENC_BUFFER_FORMAT>
    pixel_format_map = {
      // Planar YUV 4:2:0 (Y plane followed by U and V planes). Chosen over NV12 because NPP can
      // fill it from packed RGB/BGR with a single kernel launch
      {VideoCompressionType::LOSSY, NV_ENC_BUFFER_FORMAT_IYUV},
    };

  /**
   * @brief Configuration parameters for the NVENC video encoder.
   *
   * @var int buffer_length
   *   Number of input/bitstream buffer pairs the encoder cycles through.
   *
   * @var VideoCompressionType compression_type
   *   The compression mode (lossy or lossless) selected for the stream.
   *
   * @var int idr_interval
   *   Interval (in frames) between IDR (Instantaneous Decoder Refresh) keyframes.
   *
   * @var int i_frame_interval
   *   Interval (in frames) between I-frames, that is, the GOP length.
   *
   * @var int frame_rate_numerator
   *   Numerator of the target frame rate (numerator in second).
   *
   * @var int frame_rate_denominator
   *   Denominator of the target frame rate (denominator in frames).
   *
   * @var GUID preset_guid
   *   NVENC encode preset that tunes the encoder for speed or quality.
   *
   * @var NV_ENC_TUNING_INFO tuning_info
   *   Aspect (quality or latency) the chosen preset is biased to.
   *
   * @var double target_bits_per_pixel
   *   Target bitrate expressed as bits per pixel. This value is used by the encoder to
   *   determine the target bit rate, which mainly affects encoded image quality and payload
   *   size.
   *
   * @var int gpu_id
   *   Index of the CUDA device that runs the encoding.
   */
  struct EncoderParameter
  {
    int buffer_length;
    VideoCompressionType compression_type;
    int idr_interval;
    int i_frame_interval;
    int frame_rate_numerator;
    int frame_rate_denominator;
    GUID preset_guid;
    NV_ENC_TUNING_INFO tuning_info;
    double target_bits_per_pixel;
    int gpu_id;
  };

  /**
   * @brief Constructor
   *
   * The encoder session is not created here but on the first process() call, so that the
   * parameters can still be overridden after the construction (see
   * accelerated_image_processor::ros::fetch_parameters())
   */
  explicit NvencVideoCompressor(
    NvencSupportedCodec codec, common::ParameterMap dedicated_parameters = {})
  : VideoCompressor(
      CompressorBackend::NVENC, dedicated_parameters +=
                                {{"buffer_length", static_cast<int>(4)},
                                 {"hw_preset_type", static_cast<std::string>("medium")},
                                 {"tuning_info", static_cast<std::string>("ultra_low_latency")},
                                 {"target_bits_per_pixel", static_cast<double>(0.1)},
                                 {"gpu_id", static_cast<int>(0)}}),
    codec_(codec)
  {
  }

  /**
   * @brief Destructor: clean up the resources accordingly
   */
  ~NvencVideoCompressor() override;

  /**
   * @brief [override] Check the encoder is ready to run processing.
   */
  bool is_ready() const override { return state_ != State::ERROR; }

  /**
   * @brief [override] Validate the requested compression type against the hardware capability
   *
   * Opens the encoder session if it has not been opened yet, because the capability query is
   * only available on a live session
   */
  std::tuple<bool, std::string> validate_compression_type_compatibility() override;

protected:
  enum class State : uint8_t { UNINITIALIZED, READY, ERROR };

  /**
   * @brief Encoded payload location within the locked bitstream buffer
   */
  struct PayloadInfo
  {
    const uint8_t * payload_ptr;
    size_t payload_size;
  };

  State state_{State::UNINITIALIZED};

  /**
   * @brief GUID of the codec the derived class encodes to
   */
  virtual GUID codec_guid() const = 0;

  /**
   * @brief codec dedicated parameter collection
   */
  virtual EncResult collect_codec_params_impl(
    [[maybe_unused]] const EncoderParameter & general_params) = 0;

  /**
   * @brief codec dedicated initialization steps
   *
   * Called after the preset default configuration and the common configuration are applied to
   * `encode_config`, and before nvEncInitializeEncoder() consumes it
   */
  virtual EncResult init_codec_impl(NV_ENC_CONFIG & encode_config) = 0;

  /**
   * @brief Payload preprocessing implementation
   * Some codecs may need dedicated handling for the encoded payload, which this function handles
   */
  virtual PayloadInfo payload_preprocess_impl(const NV_ENC_LOCK_BITSTREAM & locked_bitstream)
  {
    return {
      reinterpret_cast<const uint8_t *>(locked_bitstream.bitstreamBufferPtr),
      static_cast<size_t>(locked_bitstream.bitstreamSizeInBytes)};
  }

  /**
   * @brief Codec dedicated payload copy implementation
   * Similar to payload preprocess, this function handles codec dedicated data copy
   */
  virtual void payload_copy_impl(
    [[maybe_unused]] const bool is_keyframe, const PayloadInfo & payload_info,
    std::vector<uint8_t> & copy_destination)
  {
    const auto & [payload_ptr, payload_size] = payload_info;
    copy_destination.resize(payload_size);
    std::memcpy(copy_destination.data(), payload_ptr, payload_size);
  }

  /**
   * @brief Collects and validates encoder parameters from the compressor's dedicated parameter
   *        map.
   *
   * @param params Reference to an `EncoderParameter` struct that will be populated with the
   *               collected values.
   *
   * @return An `EncResult` indicating success or failure.
   */
  EncResult collect_params(EncoderParameter & params);

  /**
   * @brief Load NvEncodeAPI, prepare the CUDA context and open an encode session
   *
   * Does nothing when the session has already been opened
   */
  EncResult ensure_session(const int gpu_id);

  EncoderParameter encoder_params_;

private:
  /**
   * @brief Buffer set the encoder cycles through, one per frame in flight
   */
  struct FrameBuffer
  {
    //! Device memory holding the color converted frame (planar YUV 4:2:0)
    uint8_t * yuv_device{nullptr};
    //! Row pitch in bytes of the luma plane. The chroma planes use the half of it
    size_t yuv_pitch{0};
    //! Handle of `yuv_device` registered to the encoder as an input resource
    NV_ENC_REGISTERED_PTR registered_resource{nullptr};
    //! Bitstream buffer the encoder writes the encoded payload to (= output buffer)
    NV_ENC_OUTPUT_PTR bitstream_buffer{nullptr};
  };

  common::Image process_impl(const common::Image & image) override;
  EncResult load_nvenc_api();
  EncResult init_cuda(const int gpu_id);
  EncResult init_encoder(const common::Image & image);
  EncResult setup_buffers(const uint32_t width, const uint32_t height);
  EncResult convert_to_yuv(const common::Image & image, const FrameBuffer & buffer);
  EncResult encode(const common::Image & image, FrameBuffer & buffer, common::Image & encoded);
  void release_resources();
  inline EncStatus record_error(const std::string & msg)
  {
    last_error_ = msg;
    state_ = State::ERROR;
    std::cerr << msg << std::endl;
    return EncStatus(false, msg);
  }

  std::string last_error_{""};
  NvencSupportedCodec codec_;

  //! Handle of `libnvidia-encode.so.1`, which the NVIDIA display driver provides
  void * nvenc_library_{nullptr};
  //! NvEncodeAPI function table
  NV_ENCODE_API_FUNCTION_LIST nvenc_{};
  //! Encode session handle
  void * encoder_session_{nullptr};
  //! CUDA context the encode session belongs to. Owned by the CUDA runtime, not by this class
  CUcontext cu_context_{nullptr};
  cudaStream_t stream_{};
  NppStreamContext npp_stream_ctx_{};

  //! Device memory holding the packed RGB/BGR source image transferred from the host
  uint8_t * rgb_device_{nullptr};
  size_t rgb_pitch_{0};

  std::vector<FrameBuffer> frame_buffers_{};
  size_t next_buffer_index_{0};

  uint32_t encode_width_{0};
  uint32_t encode_height_{0};
  NV_ENC_BUFFER_FORMAT buffer_format_{NV_ENC_BUFFER_FORMAT_IYUV};
  uint64_t next_pts_{0};
};
#endif  // NVENC_AVAILABLE
}  // namespace accelerated_image_processor::compression
