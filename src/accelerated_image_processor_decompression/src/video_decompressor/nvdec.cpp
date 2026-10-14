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

#include "accelerated_image_processor_decompression/video_decompressor.hpp"
#include "color_convert.hpp"

#include <accelerated_image_processor_common/datatype.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ffnvcodec/dynlink_cuviddec.h>
#include <ffnvcodec/dynlink_loader.h>
#include <ffnvcodec/dynlink_nvcuvid.h>
#include <nppi_color_conversion.h>
#include <nppi_support_functions.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace accelerated_image_processor::decompression
{
namespace
{
std::string cuda_driver_error(CUresult status)
{
  const char * name = nullptr;
  const char * description = nullptr;
  cuGetErrorName(status, &name);
  cuGetErrorString(status, &description);

  std::string message = name ? name : "unknown CUDA driver error";
  if (description) {
    message += ": ";
    message += description;
  }
  return message;
}

void throw_on_cuda_driver_error(CUresult status, const std::string & operation)
{
  if (status != CUDA_SUCCESS) {
    throw std::runtime_error(operation + " failed: " + cuda_driver_error(status));
  }
}

void throw_on_cuda_error(cudaError_t status, const std::string & operation)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(operation + " failed: " + cudaGetErrorString(status));
  }
}

void log_cuda_driver_error(CUresult status, const char * operation)
{
  if (status != CUDA_SUCCESS) {
    std::cerr << operation << " failed: " << cuda_driver_error(status) << std::endl;
  }
}
}  // namespace

class NvDecVideoDecompressor final : public VideoDecompressor
{
public:
  explicit NvDecVideoDecompressor(common::ParameterMap dedicated_parameters = {})
  : VideoDecompressor(VideoBackend::NVDEC, dedicated_parameters)
  {
  }

  ~NvDecVideoDecompressor() override { release_resources(); }

  /**
   * Decode one encoded packet and invoke the registered postprocess for every output frame.
   * Overrides BaseProcessor::process() because one packet can yield zero or multiple frames.
   * postprocess() runs after decode_mutex_ is released, so the callback may call process() again.
   * If the same instance is called from multiple threads, the callbacks may run concurrently.
   * @return The first decoded frame, or std::nullopt if the packet produced no frame.
   */
  std::optional<common::Image> process(const common::Image & image) override
  {
    auto processed_vec = process_packet(image);

    for (auto & decoded : processed_vec) {
      this->postprocess(decoded);
    }

    if (processed_vec.empty()) {
      return std::nullopt;
    } else {
      return std::move(processed_vec.front());
    }
  }

  /**
   * Dummy override for the pure virtual function. Not used because process() is overridden.
   */
  common::Image process_impl(const common::Image & image) override { return image; }

  bool is_ready() const override
  {
    return state_ == State::UNINITIALIZED || state_ == State::READY;
  }

private:
  enum class State : uint8_t { UNINITIALIZED, READY, DRAINED, ERROR };

  struct PacketMetadata
  {
    std::string frame_id;
    int64_t timestamp;
    std::optional<uint64_t> pts;
    std::optional<uint8_t> flags;
  };

  static constexpr std::size_t kMaxPacketMetadataCount = 256;

  /**
   * Decode one encoded packet under decode_mutex_ and return the decoded frames.
   * NVDEC parser callbacks are synchronous with cuvidParseVideoData(), so decoded_images_ is
   * complete when that call returns; it is moved out so that callers can use the frames after
   * the lock is released.
   * @return The decoded frames, or an empty vector if none was produced or an error occurred.
   */
  std::vector<common::Image> process_packet(const common::Image & image)
  {
    // Do not invoke postprocess() while holding this lock: it deadlocks if the callback calls
    // process() again. process() invokes it after this function returns.
    std::lock_guard<std::mutex> lock(decode_mutex_);

    try {
      if (state_ == State::ERROR) {
        throw std::runtime_error(last_error_);
      }
      if (state_ == State::DRAINED) {
        if (image.data.empty()) {
          return std::vector<common::Image>();
        }
        throw std::runtime_error("The NVDEC stream was already drained");
      }
      if (state_ == State::UNINITIALIZED && image.data.empty()) {
        return std::vector<common::Image>();
      }
      if (state_ == State::UNINITIALIZED) {
        initialize(image.format);
      }
      const bool is_end_of_stream = image.data.empty();
      if (!is_end_of_stream && image.format != input_format_) {
        throw std::runtime_error("The encoded video format changed after NVDEC initialization");
      }

      // Joining an AV1 stream mid-GOP yields packets without OBU_SEQUENCE_HEADER, which NVDEC
      // cannot parse (it reports a garbage CUVIDEOFORMAT with coded_width == 0). Drop inter frames
      // until a key frame (which carries the sequence header) establishes the decoder. A missing
      // flag is treated as "not a key frame", so we keep waiting until one arrives.
      const bool is_keyframe = image.flags.has_value() && (image.flags.value() & 0x1) != 0;
      if (!decoder_ && !is_end_of_stream && !is_keyframe) {
        return std::vector<common::Image>();
      }

      throw_on_cuda_error(cudaSetDevice(gpu_id_), "Selecting the CUDA device");

      CUVIDSOURCEDATAPACKET packet{};
      if (
        image.data.size() >
        static_cast<std::size_t>(std::numeric_limits<decltype(packet.payload_size)>::max())) {
        throw std::runtime_error("The encoded packet is too large for NVDEC");
      }

      if (is_end_of_stream) {
        packet.flags = CUVID_PKT_ENDOFSTREAM;
      } else {
        packet.payload = image.data.data();
        packet.payload_size = static_cast<decltype(packet.payload_size)>(image.data.size());
        packet.flags = CUVID_PKT_TIMESTAMP;
        packet.timestamp = next_packet_timestamp_++;

        packet_metadata_.insert_or_assign(
          packet.timestamp,
          PacketMetadata{image.frame_id, image.timestamp, image.pts, image.flags});
        packet_metadata_order_.push_back(packet.timestamp);
        while (packet_metadata_order_.size() > kMaxPacketMetadataCount) {
          packet_metadata_.erase(packet_metadata_order_.front());
          packet_metadata_order_.pop_front();
        }
      }

      decoded_images_.clear();
      callback_error_.front() = '\0';
      const CUresult status = cuvid_->cuvidParseVideoData(parser_, &packet);

      if (status != CUDA_SUCCESS) {
        const std::string detail =
          callback_error_.front() == '\0' ? cuda_driver_error(status) : callback_error_.data();
        throw std::runtime_error("Parsing the encoded packet failed: " + detail);
      }
      if (is_end_of_stream) {
        state_ = State::DRAINED;
      }

      if (decoded_images_.empty()) {
        return std::vector<common::Image>();
      }
      return std::move(decoded_images_);
    } catch (...) {
      auto exception = std::current_exception();
      set_exception_ptr(exception);
      release_resources();
      try {
        std::rethrow_exception(exception);
      } catch (const std::exception & error) {
        record_error(error.what());
      } catch (...) {
        record_error("Unknown NVDEC error");
      }
      return std::vector<common::Image>();
    }
  }

  static cudaVideoCodec to_nvdec_codec(common::ImageFormat format)
  {
    switch (format) {
      case common::ImageFormat::H264:
        return cudaVideoCodec_H264;
      case common::ImageFormat::H265:
        return cudaVideoCodec_HEVC;
      case common::ImageFormat::AV1:
        return cudaVideoCodec_AV1;
      default:
        throw std::runtime_error("NVDEC supports only H264, H265, and AV1 input");
    }
  }

  void initialize(common::ImageFormat format)
  {
    input_format_ = format;
    codec_ = to_nvdec_codec(format);

    if (cuvid_load_functions(&cuvid_, nullptr) < 0) {
      throw std::runtime_error("Loading libnvcuvid through nv-codec-headers failed");
    }

    throw_on_cuda_driver_error(cuInit(0), "Initializing the CUDA driver API");

    int device_count = 0;
    throw_on_cuda_error(cudaGetDeviceCount(&device_count), "Counting CUDA devices");
    throw_on_cuda_error(cudaGetDevice(&gpu_id_), "Getting the current CUDA device");
    if (gpu_id_ < 0 || gpu_id_ >= device_count) {
      throw std::runtime_error("The current CUDA device is unavailable");
    }

    throw_on_cuda_error(cudaSetDevice(gpu_id_), "Selecting the CUDA device");
    throw_on_cuda_error(cudaStreamCreate(&stream_), "Creating a CUDA stream");
    throw_on_cuda_driver_error(
      cuStreamGetCtx(reinterpret_cast<CUstream>(stream_), &cu_context_),
      "Getting the CUDA context for the stream");

    initialize_npp_stream_context();

    throw_on_cuda_driver_error(
      cuvid_->cuvidCtxLockCreate(&context_lock_, cu_context_),
      "Creating the NVDEC CUDA context lock");

    CUVIDPARSERPARAMS parser_parameters{};
    parser_parameters.CodecType = codec_;
    parser_parameters.ulMaxNumDecodeSurfaces = 1;
    parser_parameters.ulMaxDisplayDelay = 0;
    parser_parameters.pUserData = this;
    parser_parameters.pfnSequenceCallback = handle_video_sequence_callback;
    parser_parameters.pfnDecodePicture = handle_picture_decode_callback;
    parser_parameters.pfnDisplayPicture = handle_picture_display_callback;

    throw_on_cuda_driver_error(
      cuvid_->cuvidCreateVideoParser(&parser_, &parser_parameters),
      "Creating the NVDEC video parser");
    state_ = State::READY;
  }

  void initialize_npp_stream_context()
  {
    npp_stream_context_.hStream = stream_;
    throw_on_cuda_error(
      cudaGetDevice(&npp_stream_context_.nCudaDeviceId), "Getting the current CUDA device");

    cudaDeviceProp device_properties{};
    throw_on_cuda_error(
      cudaGetDeviceProperties(&device_properties, npp_stream_context_.nCudaDeviceId),
      "Getting CUDA device properties");
    npp_stream_context_.nMultiProcessorCount = device_properties.multiProcessorCount;
    npp_stream_context_.nMaxThreadsPerMultiProcessor =
      device_properties.maxThreadsPerMultiProcessor;
    npp_stream_context_.nMaxThreadsPerBlock = device_properties.maxThreadsPerBlock;
    npp_stream_context_.nSharedMemPerBlock = device_properties.sharedMemPerBlock;
    throw_on_cuda_error(
      cudaDeviceGetAttribute(
        &npp_stream_context_.nCudaDevAttrComputeCapabilityMajor, cudaDevAttrComputeCapabilityMajor,
        npp_stream_context_.nCudaDeviceId),
      "Getting the CUDA compute capability major version");
    throw_on_cuda_error(
      cudaDeviceGetAttribute(
        &npp_stream_context_.nCudaDevAttrComputeCapabilityMinor, cudaDevAttrComputeCapabilityMinor,
        npp_stream_context_.nCudaDeviceId),
      "Getting the CUDA compute capability minor version");
    throw_on_cuda_error(
      cudaStreamGetFlags(npp_stream_context_.hStream, &npp_stream_context_.nStreamFlags),
      "Getting CUDA stream flags");
  }

  template <typename T, typename Handler>
  static int callback_base(void * user_data, T arg, Handler handler, const char * context) noexcept
  {
    auto * self = static_cast<NvDecVideoDecompressor *>(user_data);
    if (!self || !arg) {
      return 0;
    }
    try {
      return (self->*handler)(arg);
    } catch (const std::exception & error) {
      return self->record_callback_error(error.what());
    } catch (...) {
      char msg[128];
      std::snprintf(msg, sizeof(msg), "Unknown error in the %s", context);
      return self->record_callback_error(msg);
    }
  }

  static int CUDAAPI
  handle_video_sequence_callback(void * user_data, CUVIDEOFORMAT * format) noexcept
  {
    return callback_base(
      user_data, format, &NvDecVideoDecompressor::handle_video_sequence, "NVDEC sequence callback");
  }

  static int CUDAAPI
  handle_picture_decode_callback(void * user_data, CUVIDPICPARAMS * picture_parameters) noexcept
  {
    return callback_base(
      user_data, picture_parameters, &NvDecVideoDecompressor::handle_picture_decode,
      "NVDEC decode callback");
  }

  static int CUDAAPI
  handle_picture_display_callback(void * user_data, CUVIDPARSERDISPINFO * display_info) noexcept
  {
    return callback_base(
      user_data, display_info, &NvDecVideoDecompressor::handle_picture_display,
      "NVDEC display callback");
  }

  int handle_video_sequence(const CUVIDEOFORMAT * format)
  {
    // This would be called whenever sequence header arrives.
    // If the decoder_ was already instantiated, reuse it as long as the passed configuration is
    // identical
    const unsigned int required_decode_surfaces =
      std::max(1U, static_cast<unsigned int>(format->min_num_decode_surfaces));
    if (decoder_) {
      const bool unchanged =
        format->codec == codec_ && format->chroma_format == chroma_format_ &&
        format->bit_depth_luma_minus8 == bit_depth_minus_8_ &&
        format->coded_width == surface_width_ && format->coded_height == surface_height_ &&
        format->display_area.left == display_left_ && format->display_area.top == display_top_ &&
        format->display_area.right - format->display_area.left == static_cast<int>(output_width_) &&
        format->display_area.bottom - format->display_area.top == static_cast<int>(output_height_);
      if (!unchanged || required_decode_surfaces > decode_surface_count_) {
        return record_callback_error("Dynamic NVDEC video format changes are unsupported");
      }
      return static_cast<int>(decode_surface_count_);
    }

    if (format->bit_depth_luma_minus8 != 0) {
      return record_callback_error("NVDEC high-bit-depth output is unsupported");
    }

    switch (format->chroma_format) {
      case cudaVideoChromaFormat_420:
        output_surface_format_ = cudaVideoSurfaceFormat_NV12;
        break;
      case cudaVideoChromaFormat_444:
        output_surface_format_ = cudaVideoSurfaceFormat_YUV444;
        break;
      default:
        return record_callback_error("NVDEC supports only 4:2:0 and 4:4:4 chroma input");
    }

    if (!cuvid_->cuvidGetDecoderCaps) {
      // Since this API is declared as optional, check if it is available
      return record_callback_error("cuvidGetDecoderCaps is unavailable in the NVIDIA driver");
    }

    CUVIDDECODECAPS capabilities{};
    capabilities.eCodecType = format->codec;
    capabilities.eChromaFormat = format->chroma_format;
    capabilities.nBitDepthMinus8 = format->bit_depth_luma_minus8;

    // Get decoder capability and check if the specified configuration is supported
    const CUresult caps_status = cuvid_->cuvidGetDecoderCaps(&capabilities);
    if (caps_status != CUDA_SUCCESS) {
      return record_callback_cuda_error(caps_status, "Querying NVDEC capabilities");
    }
    if (!capabilities.bIsSupported) {
      return record_callback_error("The GPU does not support this NVDEC video format");
    }

    // nOutputFormatMask: bit N == support for cudaVideoSurfaceFormat value N (NVDEC contract)
    const auto output_format_bit = 1U << static_cast<unsigned int>(output_surface_format_);
    if ((capabilities.nOutputFormatMask & output_format_bit) == 0) {
      return record_callback_error("The GPU does not support the required NVDEC output format");
    }
    if (
      format->coded_width < capabilities.nMinWidth ||
      format->coded_height < capabilities.nMinHeight ||
      capabilities.nMaxWidth < format->coded_width ||
      capabilities.nMaxHeight < format->coded_height) {
      return record_callback_error("The encoded resolution is outside the NVDEC capability range");
    }

    // NVDEC caps contract: CodedWidth*CodedHeight/256 must be <= nMaxMBCount (256 = 16x16-px
    // macroblock).
    const uint64_t macroblock_count =
      static_cast<uint64_t>(format->coded_width) * format->coded_height / 256U;
    if (macroblock_count > capabilities.nMaxMBCount) {
      return record_callback_error("The encoded resolution exceeds the NVDEC macroblock limit");
    }

    const int display_width = format->display_area.right - format->display_area.left;
    const int display_height = format->display_area.bottom - format->display_area.top;
    if (
      format->display_area.left < 0 || format->display_area.top < 0 || display_width <= 0 ||
      display_height <= 0 || format->display_area.right > static_cast<int>(format->coded_width) ||
      format->display_area.bottom > static_cast<int>(format->coded_height)) {
      return record_callback_error("NVDEC reported an invalid display area");
    }
    // Defensive overflow guard for the int casts and width*3 (BGR) below.
    // Already bounded by the nMaxWidth/nMaxHeight caps check above, so normally unreachable.
    if (
      display_width > std::numeric_limits<int>::max() / 3 ||
      format->coded_width > static_cast<unsigned int>(std::numeric_limits<int>::max()) ||
      format->coded_height > static_cast<unsigned int>(std::numeric_limits<int>::max())) {
      return record_callback_error("The decoded resolution is too large");
    }
    // 4:2:0 chroma is half-res (2x2), so crop offset/size must be even. (x|y|w|h)&1 => any is odd.
    if (
      format->chroma_format == cudaVideoChromaFormat_420 &&
      ((format->display_area.left | format->display_area.top | display_width | display_height) &
       1)) {
      return record_callback_error("NVDEC 4:2:0 display dimensions must be even");
    }

    CUVIDDECODECREATEINFO create_info{};
    create_info.CodecType = format->codec;
    create_info.ChromaFormat = format->chroma_format;
    create_info.OutputFormat = output_surface_format_;
    create_info.bitDepthMinus8 = format->bit_depth_luma_minus8;
    create_info.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
    create_info.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    create_info.ulWidth = format->coded_width;    // in pixel
    create_info.ulHeight = format->coded_height;  // in pixel
    create_info.ulMaxWidth = format->coded_width;
    create_info.ulMaxHeight = format->coded_height;
    create_info.ulNumDecodeSurfaces = required_decode_surfaces;
    create_info.ulNumOutputSurfaces = 2;
    create_info.ulTargetWidth = format->coded_width;
    create_info.ulTargetHeight = format->coded_height;
    create_info.vidLock = context_lock_;

    const CUresult create_status = cuvid_->cuvidCreateDecoder(&decoder_, &create_info);
    if (create_status != CUDA_SUCCESS) {
      return record_callback_cuda_error(create_status, "Creating the NVDEC decoder");
    }

    // Save the current configuration so that configuration identity on the next call
    chroma_format_ = format->chroma_format;
    bit_depth_minus_8_ = format->bit_depth_luma_minus8;
    surface_width_ = format->coded_width;
    surface_height_ = format->coded_height;
    display_left_ = format->display_area.left;
    display_top_ = format->display_area.top;
    output_width_ = static_cast<uint32_t>(display_width);
    output_height_ = static_cast<uint32_t>(display_height);
    decode_surface_count_ = required_decode_surfaces;

    // Remember the signalled matrix coefficients so the YUV->BGR conversion can match the encoder.
    // We do NOT read video_full_range_flag: cuvid does not parse the AV1 color_range bit (always
    // reports 0), and the correct NPP decode variant is instead chosen empirically to invert the
    // encoder (verified by a round-trip test; see README and the conversion below).
    matrix_coefficients_ = format->video_signal_description.matrix_coefficients;

    return static_cast<int>(decode_surface_count_);
  }

  int handle_picture_decode(CUVIDPICPARAMS * picture_parameters)
  {
    if (!decoder_) {
      return record_callback_error("NVDEC decoder is not initialized");
    }
    const CUresult status = cuvid_->cuvidDecodePicture(decoder_, picture_parameters);
    if (status != CUDA_SUCCESS) {
      return record_callback_cuda_error(status, "Decoding an NVDEC picture");
    }
    return 1;
  }

  int handle_picture_display(const CUVIDPARSERDISPINFO * display_info)
  {
    if (!decoder_) {
      return record_callback_error("NVDEC display callback has no initialized decoder");
    }
    const auto metadata_iterator = packet_metadata_.find(display_info->timestamp);
    if (metadata_iterator == packet_metadata_.end()) {
      return record_callback_error("NVDEC display callback has no matching packet metadata");
    }
    if (!prepare_bgr_buffer()) {
      return 0;
    }

    const auto & metadata = metadata_iterator->second;
    common::Image output{};
    output.frame_id = metadata.frame_id;
    output.timestamp = metadata.timestamp;
    output.width = output_width_;
    output.height = output_height_;
    output.step = output_width_ * 3U;
    output.format = common::ImageFormat::RAW;
    output.encoding = common::ImageEncoding::BGR;
    output.is_bigendian = false;
    output.pts = metadata.pts;
    output.flags = metadata.flags;
    output.data.resize(static_cast<std::size_t>(output.step) * output.height);

    CUVIDPROCPARAMS processing_parameters{};
    processing_parameters.progressive_frame = display_info->progressive_frame;
    // From NVIDIA NvDecoder sample: maps repeat_first_field (-1..2) to second_field (0..3).
    // This value will be used if input is interrace/telecine format, otherwise
    // (progressive_frame==1), ignored
    processing_parameters.second_field = display_info->repeat_first_field + 1;
    processing_parameters.top_field_first = display_info->top_field_first;
    processing_parameters.unpaired_field = display_info->repeat_first_field < 0;
    processing_parameters.output_stream = reinterpret_cast<CUstream>(stream_);

    // Map frame on the GPU memory
    CUdeviceptr mapped_frame = 0;
    unsigned int source_pitch = 0;
    const CUresult map_status = cuvid_->cuvidMapVideoFrame(
      decoder_, display_info->picture_index, &mapped_frame, &source_pitch, &processing_parameters);
    if (map_status != CUDA_SUCCESS) {
      return record_callback_cuda_error(map_status, "Mapping an NVDEC output frame");
    }

    const NppiSize roi{static_cast<int>(output_width_), static_cast<int>(output_height_)};
    // NVDEC maps the full "coded surface" (surface_width_ x surface_height_, i.e. coded_width/
    // height padded to macroblock alignment), not the picture to show. The visible "display area"
    // is a sub-rectangle at offset (display_left_, display_top_) with size output_width_ x
    // output_height_. So each plane pointer must be advanced from the coded-surface origin
    // (mapped_frame) to the display/crop origin before the ROI-sized NPP conversion.
    const CUdeviceptr y_plane_address =
      mapped_frame + static_cast<CUdeviceptr>(display_top_) * source_pitch + display_left_;
    const auto * y_plane =
      reinterpret_cast<const Npp8u *>(static_cast<std::uintptr_t>(y_plane_address));

    // Pick the YUV->BGR inverse that matches the encoder for the stream's matrix coefficients.
    // matrix_coefficients per ITU-T H.273: 1 = BT.709, 5/6 = BT.601; anything else
    // (including 2 = unspecified) defaults to BT.709, this project's encoder convention.
    const bool is_bt601 = matrix_coefficients_ == 5 || matrix_coefficients_ == 6;
    NppStatus conversion_status = NPP_NOT_SUPPORTED_MODE_ERROR;
    if (output_surface_format_ == cudaVideoSurfaceFormat_NV12) {
      // UV plane follows the full luma plane (surface_height_ rows). 4:2:0 chroma is half-res,
      // hence display_top_/2 (display dims were validated to be even above).
      const CUdeviceptr uv_plane_address =
        mapped_frame + static_cast<CUdeviceptr>(surface_height_) * source_pitch +
        static_cast<CUdeviceptr>(display_top_ / 2) * source_pitch + display_left_;
      const Npp8u * source_planes[2] = {
        y_plane, reinterpret_cast<const Npp8u *>(static_cast<std::uintptr_t>(uv_plane_address))};
      if (is_bt601) {
        // BT.601 (limited range); NPP has no full-range 4:2:0 BT.601 variant.
        conversion_status = nppiNV12ToBGR_8u_P2C3R_Ctx(
          source_planes, static_cast<int>(source_pitch), bgr_device_, bgr_pitch_, roi,
          npp_stream_context_);
      } else {
        // BT.709. NPP's fixed NV12->BGR converters do not correctly invert the encoder's
        // full-range nppiBGRToYCbCr420_709HDTV (709HDTV desaturates chroma, 709CSC crushes the
        // luma range), so apply the standard ITU-R BT.709 full-range matrix with our own kernel.
        const cudaError_t kernel_status = nv12_to_bgr_bt709_full(
          source_planes[0], source_planes[1], static_cast<int>(source_pitch), bgr_device_,
          bgr_pitch_, static_cast<int>(output_width_), static_cast<int>(output_height_), stream_);
        conversion_status =
          kernel_status == cudaSuccess ? NPP_SUCCESS : NPP_NOT_SUPPORTED_MODE_ERROR;
      }
    } else if (output_surface_format_ == cudaVideoSurfaceFormat_YUV444) {
      const CUdeviceptr u_plane_address =
        mapped_frame + static_cast<CUdeviceptr>(surface_height_) * source_pitch +
        static_cast<CUdeviceptr>(display_top_) * source_pitch + display_left_;
      const CUdeviceptr v_plane_address =
        mapped_frame + static_cast<CUdeviceptr>(surface_height_) * source_pitch * 2U +
        static_cast<CUdeviceptr>(display_top_) * source_pitch + display_left_;
      const Npp8u * source_planes[3] = {
        y_plane, reinterpret_cast<const Npp8u *>(static_cast<std::uintptr_t>(u_plane_address)),
        reinterpret_cast<const Npp8u *>(static_cast<std::uintptr_t>(v_plane_address))};
      if (is_bt601) {
        // BT.601 full range.
        conversion_status = nppiYCbCrToBGR_8u_P3C3R_Ctx(
          source_planes, static_cast<int>(source_pitch), bgr_device_, bgr_pitch_, roi,
          npp_stream_context_);
      } else {
        // BT.709: same standard ITU-R BT.709 full-range matrix as the NV12 path, via our kernel.
        // The 4:4:4 path is not exercised by the 4:2:0 pipeline but is kept consistent.
        const cudaError_t kernel_status = yuv444_to_bgr_bt709_full(
          source_planes[0], source_planes[1], source_planes[2], static_cast<int>(source_pitch),
          bgr_device_, bgr_pitch_, static_cast<int>(output_width_),
          static_cast<int>(output_height_), stream_);
        conversion_status =
          kernel_status == cudaSuccess ? NPP_SUCCESS : NPP_NOT_SUPPORTED_MODE_ERROR;
      }
    }

    if (conversion_status != NPP_SUCCESS) {
      cudaStreamSynchronize(stream_);
      cuvid_->cuvidUnmapVideoFrame(decoder_, mapped_frame);
      return record_callback_error(
        "Converting the NVDEC output to BGR failed with NPP status " +
        std::to_string(conversion_status));
    }

    const cudaError_t copy_status = cudaMemcpy2DAsync(
      output.data.data(), output.step, bgr_device_, bgr_pitch_, output.step, output.height,
      cudaMemcpyDeviceToHost, stream_);
    if (copy_status != cudaSuccess) {
      cudaStreamSynchronize(stream_);
      cuvid_->cuvidUnmapVideoFrame(decoder_, mapped_frame);
      return record_callback_error(
        std::string("Copying the decoded BGR frame to the host failed: ") +
        cudaGetErrorString(copy_status));
    }

    const cudaError_t synchronize_status = cudaStreamSynchronize(stream_);
    if (synchronize_status != cudaSuccess) {
      cuvid_->cuvidUnmapVideoFrame(decoder_, mapped_frame);
      return record_callback_error(
        std::string("Synchronizing the NVDEC output stream failed: ") +
        cudaGetErrorString(synchronize_status));
    }

    const CUresult unmap_status = cuvid_->cuvidUnmapVideoFrame(decoder_, mapped_frame);
    if (unmap_status != CUDA_SUCCESS) {
      return record_callback_cuda_error(unmap_status, "Unmapping an NVDEC output frame");
    }

    decoded_images_.push_back(std::move(output));
    return 1;
  }

  bool prepare_bgr_buffer()
  {
    if (bgr_device_ && bgr_width_ == output_width_ && bgr_height_ == output_height_) {
      return true;
    }

    if (bgr_device_) {
      nppiFree(bgr_device_);
      bgr_device_ = nullptr;
    }

    bgr_device_ = nppiMalloc_8u_C3(
      static_cast<int>(output_width_), static_cast<int>(output_height_), &bgr_pitch_);
    if (!bgr_device_) {
      bgr_pitch_ = 0;
      return record_callback_error("Allocating the decoded BGR device buffer failed") != 0;
    }

    bgr_width_ = output_width_;
    bgr_height_ = output_height_;
    return true;
  }

  int record_callback_cuda_error(CUresult status, const std::string & operation) noexcept
  {
    try {
      return record_callback_error(operation + " failed: " + cuda_driver_error(status));
    } catch (...) {
      return record_callback_error("An NVDEC CUDA driver call failed");
    }
  }

  int record_callback_error(const std::string & message) noexcept
  {
    return record_callback_error(message.c_str());
  }

  int record_callback_error(const char * message) noexcept
  {
    std::snprintf(
      callback_error_.data(), callback_error_.size(), "%s",
      message ? message : "Unknown NVDEC callback error");
    return 0;
  }

  void record_error(std::string message)
  {
    last_error_ = std::move(message);
    state_ = State::ERROR;
    std::cerr << last_error_ << std::endl;
  }

  void release_resources() noexcept
  {
    if (gpu_id_ >= 0) {
      const cudaError_t status = cudaSetDevice(gpu_id_);
      if (status != cudaSuccess) {
        std::cerr << "Selecting the CUDA device during NVDEC cleanup failed: "
                  << cudaGetErrorString(status) << std::endl;
      }
    }

    if (cuvid_ && parser_) {
      log_cuda_driver_error(cuvid_->cuvidDestroyVideoParser(parser_), "Destroying NVDEC parser");
      parser_ = nullptr;
    }
    if (cuvid_ && decoder_) {
      log_cuda_driver_error(cuvid_->cuvidDestroyDecoder(decoder_), "Destroying NVDEC decoder");
      decoder_ = nullptr;
    }
    if (cuvid_ && context_lock_) {
      log_cuda_driver_error(
        cuvid_->cuvidCtxLockDestroy(context_lock_), "Destroying NVDEC context lock");
      context_lock_ = nullptr;
    }

    if (bgr_device_) {
      nppiFree(bgr_device_);
      bgr_device_ = nullptr;
    }
    if (stream_) {
      const cudaError_t status = cudaStreamDestroy(stream_);
      if (status != cudaSuccess) {
        std::cerr << "Destroying the NVDEC CUDA stream failed: " << cudaGetErrorString(status)
                  << std::endl;
      }
      stream_ = nullptr;
    }

    cuvid_free_functions(&cuvid_);
    cu_context_ = nullptr;
  }

  State state_{State::UNINITIALIZED};
  std::string last_error_{};
  std::array<char, 512> callback_error_{};

  common::ImageFormat input_format_{common::ImageFormat::RAW};
  cudaVideoCodec codec_{cudaVideoCodec_NumCodecs};
  cudaVideoChromaFormat chroma_format_{cudaVideoChromaFormat_Monochrome};
  cudaVideoSurfaceFormat output_surface_format_{cudaVideoSurfaceFormat_NV12};
  unsigned int bit_depth_minus_8_{0};
  uint8_t matrix_coefficients_{2};  // ITU-T H.273; 2 = unspecified (treated as BT.709)

  CuvidFunctions * cuvid_{nullptr};
  CUvideoparser parser_{nullptr};
  CUvideodecoder decoder_{nullptr};
  CUvideoctxlock context_lock_{nullptr};
  CUcontext cu_context_{nullptr};
  int gpu_id_{-1};

  cudaStream_t stream_{nullptr};
  NppStreamContext npp_stream_context_{};
  Npp8u * bgr_device_{nullptr};
  int bgr_pitch_{0};
  uint32_t bgr_width_{0};
  uint32_t bgr_height_{0};

  uint32_t surface_width_{0};
  uint32_t surface_height_{0};
  int display_left_{0};
  int display_top_{0};
  uint32_t output_width_{0};
  uint32_t output_height_{0};
  unsigned int decode_surface_count_{0};

  std::unordered_map<CUvideotimestamp, PacketMetadata> packet_metadata_{};
  std::deque<CUvideotimestamp> packet_metadata_order_{};
  std::vector<common::Image> decoded_images_{};
  CUvideotimestamp next_packet_timestamp_{0};

  std::mutex decode_mutex_;
};

std::unique_ptr<VideoDecompressor> make_nvdec_video_decompressor()
{
  return std::make_unique<NvDecVideoDecompressor>();
}
}  // namespace accelerated_image_processor::decompression
