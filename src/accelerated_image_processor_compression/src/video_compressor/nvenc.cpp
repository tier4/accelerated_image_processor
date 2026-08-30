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

#include "nvenc.hpp"

#include "accelerated_image_processor_compression/video_compressor.hpp"
#include "enc_result.hpp"
#include "nvenc_error_helper.hpp"

#include <accelerated_image_processor_common/helper.hpp>

#include <dlfcn.h>
#include <endian.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace
{
constexpr uint8_t AV_PKT_FLAG_KEY = 0x0001;

constexpr bool is_big_endian = (__BYTE_ORDER__ == __BIG_ENDIAN);

//! Shared library the NVIDIA display driver installs, which provides NvEncodeAPI
constexpr const char * nvenc_library_name = "libnvidia-encode.so.1";

//! Peak bitrate is set to this ratio of the target bitrate, as the Jetson backend does
constexpr double peak_bitrate_ratio = 1.2;
}  // namespace

namespace accelerated_image_processor::compression
{
#ifdef NVENC_AVAILABLE

NvencVideoCompressor::~NvencVideoCompressor()
{
  release_resources();
}

EncResult NvencVideoCompressor::collect_params(EncoderParameter & params)
{
  params.buffer_length = this->parameter_value<int>("buffer_length");
  params.compression_type = string_to_enum<VideoCompressionType>(
    this->parameter_value<std::string>("compression_type"), video_compression_type_map);

  params.idr_interval = this->parameter_value<int>("idr_frame_interval");
  params.i_frame_interval = this->parameter_value<int>("i_frame_interval");
  params.frame_rate_numerator = this->parameter_value<int>("frame_rate_numerator");
  params.frame_rate_denominator = this->parameter_value<int>("frame_rate_denominator");
  params.preset_guid =
    string_to_enum<GUID>(this->parameter_value<std::string>("hw_preset_type"), hardware_preset_map);
  params.tuning_info = string_to_enum<NV_ENC_TUNING_INFO>(
    this->parameter_value<std::string>("tuning_info"), tuning_info_map);
  params.target_bits_per_pixel = this->parameter_value<double>("target_bits_per_pixel");
  params.gpu_id = this->parameter_value<int>("gpu_id");

  if (params.buffer_length < 1) {
    return EncResult(EncStatus(false, "buffer_length must be equal to or larger than 1"));
  }
  if (params.idr_interval < 1 || params.i_frame_interval < 1) {
    return EncResult(
      EncStatus(false, "idr_frame_interval and i_frame_interval must be larger than 0"));
  }
  if (params.frame_rate_numerator < 1 || params.frame_rate_denominator < 1) {
    return EncResult(
      EncStatus(false, "frame_rate_numerator and frame_rate_denominator must be larger than 0"));
  }
  if (params.target_bits_per_pixel <= 0.0) {
    return EncResult(EncStatus(false, "target_bits_per_pixel must be larger than 0"));
  }

  return EncResult::success();
}

EncResult NvencVideoCompressor::load_nvenc_api()
{
  if (nvenc_library_) return EncResult::success();

  // NvEncodeAPI is provided by the display driver, hence it is resolved at runtime so that this
  // library stays loadable on a host that has no NVIDIA driver installed
  void * library = dlopen(nvenc_library_name, RTLD_LAZY);
  if (!library) {
    return EncResult(EncStatus(
      false, std::string("Failed to load ") + nvenc_library_name + " (" + dlerror() +
               "). Confirm the NVIDIA display driver is installed"));
  }

  // Confirm the driver is new enough for the NvEncodeAPI version this library was compiled with
  using NvEncodeAPIGetMaxSupportedVersion_t = NVENCSTATUS(NVENCAPI *)(uint32_t *);
  auto get_max_supported_version = reinterpret_cast<NvEncodeAPIGetMaxSupportedVersion_t>(
    dlsym(library, "NvEncodeAPIGetMaxSupportedVersion"));
  if (!get_max_supported_version) {
    dlclose(library);
    return EncResult(EncStatus(
      false,
      std::string("Failed to find NvEncodeAPIGetMaxSupportedVersion in ") + nvenc_library_name));
  }

  uint32_t max_version = 0;
  if (const auto stat = get_max_supported_version(&max_version); stat != NV_ENC_SUCCESS) {
    dlclose(library);
    return EncResult(EncStatus(
      false,
      "Failed to get max supported version (" + std::string(nvencstatus_to_string(stat)) + ")"));
  }
  constexpr uint32_t compiled_api_version = (NVENCAPI_MAJOR_VERSION << 4) | NVENCAPI_MINOR_VERSION;
  if (max_version < compiled_api_version) {
    const std::string msg =
      "The NvEncodeAPI version the installed driver supports (" + std::to_string(max_version >> 4) +
      "." + std::to_string(max_version & 0xF) +
      ") is older than the nv-codec-headers version this library was compiled with (" +
      std::to_string(NVENCAPI_MAJOR_VERSION) + "." + std::to_string(NVENCAPI_MINOR_VERSION) +
      "). Please consider upgrading the display driver or using the proper version of "
      "nv-codec-headers";
    dlclose(library);
    return EncResult(EncStatus(false, msg));
  }

  using NvEncodeAPICreateInstance_t = NVENCSTATUS(NVENCAPI *)(NV_ENCODE_API_FUNCTION_LIST *);
  auto create_instance =
    reinterpret_cast<NvEncodeAPICreateInstance_t>(dlsym(library, "NvEncodeAPICreateInstance"));
  if (!create_instance) {
    dlclose(library);
    return EncResult(EncStatus(
      false, std::string("Failed to find NvEncodeAPICreateInstance in ") + nvenc_library_name));
  }

  nvenc_ = {};
  nvenc_.version = NV_ENCODE_API_FUNCTION_LIST_VER;
  if (const auto stat = create_instance(&nvenc_); stat != NV_ENC_SUCCESS) {
    dlclose(library);
    return EncResult(EncStatus(
      false, "Failed to create the NvEncodeAPI instance (" +
               std::string(nvencstatus_to_string(stat)) + ")"));
  }

  nvenc_library_ = library;
  return EncResult::success();
}

EncResult NvencVideoCompressor::init_cuda(const int gpu_id)
{
  if (cu_context_) return EncResult::success();

  NVENC_CHECK_CU(cuInit(0), "Failed to initialize the CUDA driver API");

  int device_count = 0;
  NVENC_CHECK_CUDA(cudaGetDeviceCount(&device_count), "Failed to count the CUDA devices");
  if (gpu_id < 0 || gpu_id >= device_count) {
    return EncResult(EncStatus(
      false, "gpu_id " + std::to_string(gpu_id) + " is out of range (" +
               std::to_string(device_count) + " CUDA device(s) found)"));
  }

  NVENC_CHECK_CUDA(cudaSetDevice(gpu_id), "Failed to select the CUDA device");
  NVENC_CHECK_CUDA(cudaStreamCreate(&stream_), "Failed to create a CUDA stream");

  // NVENC takes the encode device as a CUcontext, hence pick up the context the CUDA runtime
  // has created for the stream above via the driver API
  NVENC_CHECK_CU(
    cuStreamGetCtx(reinterpret_cast<CUstream>(stream_), &cu_context_),
    "Failed to fetch the CUDA context of the stream");

  // Fill the NPP stream context so that the color conversion runs on the stream above
  {
    npp_stream_ctx_.hStream = stream_;
    NVENC_CHECK_CUDA(
      cudaGetDevice(&npp_stream_ctx_.nCudaDeviceId), "Failed to get the current CUDA device");
    cudaDeviceProp device_prop;
    NVENC_CHECK_CUDA(
      cudaGetDeviceProperties(&device_prop, npp_stream_ctx_.nCudaDeviceId),
      "Failed to get the CUDA device properties");
    npp_stream_ctx_.nMultiProcessorCount = device_prop.multiProcessorCount;
    npp_stream_ctx_.nMaxThreadsPerMultiProcessor = device_prop.maxThreadsPerMultiProcessor;
    npp_stream_ctx_.nMaxThreadsPerBlock = device_prop.maxThreadsPerBlock;
    npp_stream_ctx_.nSharedMemPerBlock = device_prop.sharedMemPerBlock;
    NVENC_CHECK_CUDA(
      cudaDeviceGetAttribute(
        &npp_stream_ctx_.nCudaDevAttrComputeCapabilityMajor, cudaDevAttrComputeCapabilityMajor,
        npp_stream_ctx_.nCudaDeviceId),
      "Failed to get the compute capability (major)");
    NVENC_CHECK_CUDA(
      cudaDeviceGetAttribute(
        &npp_stream_ctx_.nCudaDevAttrComputeCapabilityMinor, cudaDevAttrComputeCapabilityMinor,
        npp_stream_ctx_.nCudaDeviceId),
      "Failed to get the compute capability (minor)");
    NVENC_CHECK_CUDA(
      cudaStreamGetFlags(npp_stream_ctx_.hStream, &npp_stream_ctx_.nStreamFlags),
      "Failed to get the CUDA stream flags");
  }

  return EncResult::success();
}

EncResult NvencVideoCompressor::ensure_session(const int gpu_id)
{
  if (encoder_session_) return EncResult::success();

  if (auto res = load_nvenc_api(); !res.ok) {
    return res;
  }
  if (auto res = init_cuda(gpu_id); !res.ok) {
    return res;
  }

  NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS session_params{};
  session_params.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
  session_params.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
  session_params.device = static_cast<void *>(cu_context_);
  session_params.apiVersion = NVENCAPI_VERSION;

  void * session = nullptr;
  NVENC_CHECK(
    nvenc_.nvEncOpenEncodeSessionEx(&session_params, &session),
    "Failed to open an NVENC encode session");

  // Confirm the driver/GPU combination exposes the codec this compressor encodes to
  {
    uint32_t guid_count = 0;
    if (const auto stat = nvenc_.nvEncGetEncodeGUIDCount(session, &guid_count);
        stat != NV_ENC_SUCCESS) {
      nvenc_.nvEncDestroyEncoder(session);
      return EncResult(EncStatus(
        false,
        "Failed to count the supported codecs (" + std::string(nvencstatus_to_string(stat)) + ")"));
    }

    std::vector<GUID> encode_guids(guid_count);
    uint32_t returned_count = 0;
    if (const auto stat =
          nvenc_.nvEncGetEncodeGUIDs(session, encode_guids.data(), guid_count, &returned_count);
        stat != NV_ENC_SUCCESS) {
      nvenc_.nvEncDestroyEncoder(session);
      return EncResult(EncStatus(
        false,
        "Failed to list the supported codecs (" + std::string(nvencstatus_to_string(stat)) + ")"));
    }

    const auto target_guid = codec_guid();
    const bool is_supported = std::any_of(
      encode_guids.begin(), encode_guids.begin() + returned_count,
      [&](const GUID & guid) { return std::memcmp(&guid, &target_guid, sizeof(GUID)) == 0; });
    if (!is_supported) {
      nvenc_.nvEncDestroyEncoder(session);
      return EncResult(
        EncStatus(false, "This NVENC driver/GPU combination does not expose the requested codec"));
    }
  }

  encoder_session_ = session;
  return EncResult::success();
}

std::tuple<bool, std::string> NvencVideoCompressor::validate_compression_type_compatibility()
{
  // Load the latest parameters if the encoder is uninitialized; use the current config otherwise
  EncoderParameter latest_params = encoder_params_;
  if (state_ == State::UNINITIALIZED) {
    if (auto res = collect_params(latest_params); !res.ok) {
      return {false, res.status.message};
    }
    if (auto res = collect_codec_params_impl(latest_params); !res.ok) {
      return {false, res.status.message};
    }
  }

  // The capability query requires a live encode session
  if (auto res = ensure_session(latest_params.gpu_id); !res.ok) {
    return {false, res.status.message};
  }

  if (latest_params.compression_type == VideoCompressionType::LOSSLESS) {
    NV_ENC_CAPS_PARAM caps_param{};
    caps_param.version = NV_ENC_CAPS_PARAM_VER;
    caps_param.capsToQuery = NV_ENC_CAPS_SUPPORT_LOSSLESS_ENCODE;
    int is_supported = 0;
    if (const auto stat =
          nvenc_.nvEncGetEncodeCaps(encoder_session_, codec_guid(), &caps_param, &is_supported);
        stat != NV_ENC_SUCCESS) {
      return {
        false, "Failed to query the lossless encoding capability (" +
                 std::string(nvencstatus_to_string(stat)) + ")"};
    }
    if (!is_supported) {
      return {
        false, "Lossless compression is not supported by this GPU/driver for the requested codec"};
    }
  }

  if (pixel_format_map.find(latest_params.compression_type) == pixel_format_map.end()) {
    return {false, "The requested compression type is not implemented by the NVENC backend"};
  }

  return {true, ""};
}

EncResult NvencVideoCompressor::init_encoder(const common::Image & image)
{
  // gather parameters
  if (auto res = collect_params(encoder_params_); !res.ok) {
    return EncResult(record_error("Failed to correct parameters (" + res.status.message + ")"));
  }

  if (auto res = this->collect_codec_params_impl(encoder_params_); !res.ok) {
    return EncResult(
      record_error("Failed to correct codec dedicated parameters (" + res.status.message + ")"));
  }

  if (auto res = ensure_session(encoder_params_.gpu_id); !res.ok) {
    return EncResult(record_error("Failed to open an encode session (" + res.status.message + ")"));
  }

  // Confirm the given combination of parameters is valid
  if (auto [is_valid, msg] = validate_compression_type_compatibility(); !is_valid) {
    return EncResult(record_error("Invalid parameters (" + msg + ")"));
  }

  if (
    image.encoding != common::ImageEncoding::RGB && image.encoding != common::ImageEncoding::BGR) {
    return EncResult(record_error("Unsupported input encoding detected"));
  }

  // 4:2:0 subsampling halves the chroma resolution, hence odd dimensions cannot be handled
  if (image.width == 0 || image.height == 0 || image.width % 2 != 0 || image.height % 2 != 0) {
    return EncResult(record_error("Input image dimensions must be non-zero and even"));
  }

  encode_width_ = image.width;
  encode_height_ = image.height;
  buffer_format_ = pixel_format_map.at(encoder_params_.compression_type);

  NV_ENC_INITIALIZE_PARAMS init_params{};
  init_params.version = NV_ENC_INITIALIZE_PARAMS_VER;
  init_params.encodeGUID = codec_guid();
  init_params.presetGUID = encoder_params_.preset_guid;
  init_params.encodeWidth = encode_width_;
  init_params.encodeHeight = encode_height_;
  init_params.maxEncodeWidth = encode_width_;
  init_params.maxEncodeHeight = encode_height_;
  init_params.darWidth = encode_width_;
  init_params.darHeight = encode_height_;
  // rate is specified in [numerator (frames), denominator (second)] format, as the Jetson backend
  init_params.frameRateNum = static_cast<uint32_t>(encoder_params_.frame_rate_numerator);
  init_params.frameRateDen = static_cast<uint32_t>(encoder_params_.frame_rate_denominator);
  init_params.enablePTD = 1;  // let NvEncodeAPI decide the picture type of each frame
  init_params.tuningInfo = encoder_params_.tuning_info;

  // Start from the defaults of the chosen preset, then override what this package controls
  NV_ENC_PRESET_CONFIG preset_config{};
  preset_config.version = NV_ENC_PRESET_CONFIG_VER;
  preset_config.presetCfg.version = NV_ENC_CONFIG_VER;
  NVENC_CHECK(
    nvenc_.nvEncGetEncodePresetConfigEx(
      encoder_session_, init_params.encodeGUID, init_params.presetGUID, init_params.tuningInfo,
      &preset_config),
    "Failed to get the preset configuration");
  NV_ENC_CONFIG encode_config = preset_config.presetCfg;

  // Set I frame interval (GOP length)
  // I frame is self-decodable frame, which can be decoded without referring other frames
  // NOTE: A codec dedicated implementation may narrow this value down in init_codec_impl() when
  // the codec cannot express an I frame that is not an IDR frame (see NvencAV1Compressor)
  encode_config.gopLength = static_cast<uint32_t>(encoder_params_.i_frame_interval);
  // GOP pattern. 1 means IPP, that is, B-Frames are disabled for streaming compression
  encode_config.frameIntervalP = 1;

  if (encoder_params_.compression_type == VideoCompressionType::LOSSY) {
    // Enable variable rate control (VRC)
    encode_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;

    // compute the target bit rate from input streaming rate
    const double frame_rate = static_cast<double>(encoder_params_.frame_rate_numerator) /
                              static_cast<double>(encoder_params_.frame_rate_denominator);
    const auto target_bit_rate = static_cast<double>(image.height) *
                                 static_cast<double>(image.width) * frame_rate *
                                 encoder_params_.target_bits_per_pixel;
    encode_config.rcParams.averageBitRate = static_cast<uint32_t>(target_bit_rate);
    encode_config.rcParams.maxBitRate = static_cast<uint32_t>(peak_bitrate_ratio * target_bit_rate);
  }
  // Lookahead would make the encoder consume several frames before emitting the first packet,
  // which breaks the one frame in / one packet out contract of process()
  encode_config.rcParams.enableLookahead = 0;

  // Do codec specific configuration
  if (auto res = this->init_codec_impl(encode_config); !res.ok) {
    return EncResult(
      record_error("Codec specific configuration failed (" + res.status.message + ")"));
  }

  init_params.encodeConfig = &encode_config;
  NVENC_CHECK(
    nvenc_.nvEncInitializeEncoder(encoder_session_, &init_params),
    "Failed to initialize the encoder");

  if (auto res = setup_buffers(encode_width_, encode_height_); !res.ok) {
    return EncResult(
      record_error("Failed to setup the encoder buffers (" + res.status.message + ")"));
  }

  // Now, ready to process
  state_ = State::READY;
  return EncResult::success();
}

EncResult NvencVideoCompressor::setup_buffers(const uint32_t width, const uint32_t height)
{
  // Device memory the packed RGB/BGR source image is transferred to
  constexpr int channels = 3;
  NVENC_CHECK_CUDA(
    cudaMallocPitch(
      reinterpret_cast<void **>(&rgb_device_), &rgb_pitch_, static_cast<size_t>(width) * channels,
      height),
    "Failed to allocate the device memory for the source image");

  frame_buffers_.assign(static_cast<size_t>(encoder_params_.buffer_length), FrameBuffer{});
  for (auto & buffer : frame_buffers_) {
    // Planar YUV 4:2:0 occupies `pitch * height` bytes for the luma plane followed by
    // `(pitch / 2) * (height / 2)` bytes for each of the two chroma planes, which is exactly
    // `pitch * height * 3 / 2` bytes in total
    NVENC_CHECK_CUDA(
      cudaMallocPitch(
        reinterpret_cast<void **>(&buffer.yuv_device), &buffer.yuv_pitch, width,
        static_cast<size_t>(height) * 3 / 2),
      "Failed to allocate the device memory for the color converted image");
    if (buffer.yuv_pitch % 2 != 0) {
      return EncResult(EncStatus(
        false, "The allocated pitch is odd, which cannot hold the half sized chroma planes"));
    }

    NV_ENC_REGISTER_RESOURCE register_params{};
    register_params.version = NV_ENC_REGISTER_RESOURCE_VER;
    register_params.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
    register_params.width = width;
    register_params.height = height;
    register_params.pitch = static_cast<uint32_t>(buffer.yuv_pitch);
    register_params.resourceToRegister = buffer.yuv_device;
    register_params.bufferFormat = buffer_format_;
    register_params.bufferUsage = NV_ENC_INPUT_IMAGE;
    NVENC_CHECK(
      nvenc_.nvEncRegisterResource(encoder_session_, &register_params),
      "Failed to register the input buffer to the encoder");
    buffer.registered_resource = register_params.registeredResource;

    NV_ENC_CREATE_BITSTREAM_BUFFER create_bitstream_params{};
    create_bitstream_params.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
    NVENC_CHECK(
      nvenc_.nvEncCreateBitstreamBuffer(encoder_session_, &create_bitstream_params),
      "Failed to create the bitstream buffer");
    buffer.bitstream_buffer = create_bitstream_params.bitstreamBuffer;
  }

  return EncResult::success();
}

EncResult NvencVideoCompressor::convert_to_yuv(
  const common::Image & image, const FrameBuffer & buffer)
{
  constexpr int channels = 3;
  const size_t source_pitch =
    image.step != 0 ? image.step : static_cast<size_t>(image.width) * channels;

  NVENC_CHECK_CUDA(
    cudaMemcpy2DAsync(
      rgb_device_, rgb_pitch_, image.data.data(), source_pitch,
      static_cast<size_t>(image.width) * channels, image.height, cudaMemcpyHostToDevice, stream_),
    "Failed to transfer the source image to the device");

  const auto luma_pitch = static_cast<int>(buffer.yuv_pitch);
  const auto chroma_pitch = luma_pitch / 2;
  const size_t luma_size = buffer.yuv_pitch * encode_height_;
  const size_t chroma_size = static_cast<size_t>(chroma_pitch) * (encode_height_ / 2);
  Npp8u * destination_planes[3] = {
    buffer.yuv_device, buffer.yuv_device + luma_size, buffer.yuv_device + luma_size + chroma_size};
  int destination_steps[3] = {luma_pitch, chroma_pitch, chroma_pitch};
  NppiSize roi = {static_cast<int>(encode_width_), static_cast<int>(encode_height_)};

  // The `_JPEG_` variants yield full range (0-255) YCbCr, which corresponds to the extended
  // color range (NVBUF_COLOR_FORMAT_NV12_ER) the Jetson backend feeds its encoder with
  if (image.encoding == common::ImageEncoding::RGB) {
    NVENC_CHECK_NPP(
      nppiRGBToYCbCr420_JPEG_8u_C3P3R_Ctx(
        rgb_device_, static_cast<int>(rgb_pitch_), destination_planes, destination_steps, roi,
        npp_stream_ctx_),
      "Failed to convert the source image from RGB to YCbCr 4:2:0");
  } else {
    NVENC_CHECK_NPP(
      nppiBGRToYCbCr420_JPEG_8u_C3P3R_Ctx(
        rgb_device_, static_cast<int>(rgb_pitch_), destination_planes, destination_steps, roi,
        npp_stream_ctx_),
      "Failed to convert the source image from BGR to YCbCr 4:2:0");
  }

  // The encoder is driven by the host thread below, hence the asynchronous work submitted above
  // has to be completed before the frame is handed over to it
  // XXX: Since `NvEncMapInputResource` provides synchronization guarantee that any graphics work
  // submitted on the input buffer is completed before the buffer is used for encoding, this stream
  // synchronization may be able to omit
  NVENC_CHECK_CUDA(
    cudaStreamSynchronize(stream_), "Failed to synchronize the color conversion stream");

  return EncResult::success();
}

EncResult NvencVideoCompressor::encode(
  const common::Image & image, FrameBuffer & buffer, common::Image & encoded)
{
  NV_ENC_MAP_INPUT_RESOURCE map_params{};
  map_params.version = NV_ENC_MAP_INPUT_RESOURCE_VER;
  map_params.registeredResource = buffer.registered_resource;
  NVENC_CHECK(
    nvenc_.nvEncMapInputResource(encoder_session_, &map_params),
    "Failed to map the input buffer to the encoder");

  NV_ENC_PIC_PARAMS pic_params{};
  pic_params.version = NV_ENC_PIC_PARAMS_VER;
  pic_params.inputBuffer = map_params.mappedResource;
  pic_params.bufferFmt = map_params.mappedBufferFmt;
  pic_params.inputWidth = encode_width_;
  pic_params.inputHeight = encode_height_;
  pic_params.inputPitch = static_cast<uint32_t>(buffer.yuv_pitch);
  pic_params.outputBitstream = buffer.bitstream_buffer;
  pic_params.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
  pic_params.inputTimeStamp = next_pts_;

  if (const auto stat = nvenc_.nvEncEncodePicture(encoder_session_, &pic_params);
      stat != NV_ENC_SUCCESS) {
    nvenc_.nvEncUnmapInputResource(encoder_session_, map_params.mappedResource);
    return EncResult(EncStatus(
      false, "Failed to encode the frame (" + std::string(nvencstatus_to_string(stat)) + ")"));
  }

  NV_ENC_LOCK_BITSTREAM lock_bitstream_params{};
  lock_bitstream_params.version = NV_ENC_LOCK_BITSTREAM_VER;
  lock_bitstream_params.outputBitstream = buffer.bitstream_buffer;
  lock_bitstream_params.doNotWait = 0;  // block until the encoded payload is available
  if (const auto stat = nvenc_.nvEncLockBitstream(encoder_session_, &lock_bitstream_params);
      stat != NV_ENC_SUCCESS) {
    nvenc_.nvEncUnmapInputResource(encoder_session_, map_params.mappedResource);
    return EncResult(EncStatus(
      false,
      "Failed to lock the bitstream buffer (" + std::string(nvencstatus_to_string(stat)) + ")"));
  }

  // Execute preprocess for the encoded payload (some codec requires dedicated handling)
  const auto payload_info = payload_preprocess_impl(lock_bitstream_params);

  const bool is_keyframe = lock_bitstream_params.pictureType == NV_ENC_PIC_TYPE_IDR ||
                           lock_bitstream_params.pictureType == NV_ENC_PIC_TYPE_I;

  encoded.frame_id = image.frame_id;
  encoded.timestamp = image.timestamp;
  encoded.height = image.height;
  encoded.width = image.width;
  encoded.step = 0;  // 0 means this value is pointless because it's compressed
  encoded.encoding = image.encoding;
  encoded.format = supported_codec_format_map.at(codec_);
  encoded.pts = next_pts_++;
  encoded.flags = is_keyframe ? AV_PKT_FLAG_KEY : 0;
  encoded.is_bigendian = is_big_endian;

  payload_copy_impl(is_keyframe, payload_info, encoded.data);

  NVENC_CHECK(
    nvenc_.nvEncUnlockBitstream(encoder_session_, buffer.bitstream_buffer),
    "Failed to unlock the bitstream buffer");
  NVENC_CHECK(
    nvenc_.nvEncUnmapInputResource(encoder_session_, map_params.mappedResource),
    "Failed to unmap the input buffer");

  return EncResult::success();
}

common::Image NvencVideoCompressor::process_impl(const common::Image & image)
{
  if (state_ != State::READY) {
    if (!init_encoder(image).ok) {
      throw std::runtime_error("Encoder initialization failed: " + last_error_);
    }
  }

  // cudaSetDevice() takes effect on the calling thread only, hence the device has to be selected
  // again when process() is called from a thread other than the one that initialized the encoder
  CHECK_CUDA(cudaSetDevice(encoder_params_.gpu_id));

  // The encode session is bound to the resolution given at the initialization
  if (image.width != encode_width_ || image.height != encode_height_) {
    std::cerr << "Input image resolution has changed, which the encoder cannot follow" << std::endl;
    return common::Image();
  }

  auto & buffer = frame_buffers_[next_buffer_index_];
  next_buffer_index_ = (next_buffer_index_ + 1) % frame_buffers_.size();

  if (auto res = convert_to_yuv(image, buffer); !res.ok) {
    std::cerr << "Failed to prepare the encoder input: " << res.status.message << std::endl;
    return common::Image();
  }

  common::Image encoded;
  if (auto res = encode(image, buffer, encoded); !res.ok) {
    std::cerr << "Failed to encode: " << res.status.message << std::endl;
    return common::Image();
  }

  return encoded;
}

void NvencVideoCompressor::release_resources()
{
  if (encoder_session_) {
    if (state_ == State::READY) {
      // Notify the end of the stream so that the encoder flushes whatever it holds internally
      NV_ENC_PIC_PARAMS eos_params{};
      eos_params.version = NV_ENC_PIC_PARAMS_VER;
      eos_params.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
      nvenc_.nvEncEncodePicture(encoder_session_, &eos_params);
    }

    for (auto & buffer : frame_buffers_) {
      if (buffer.registered_resource) {
        nvenc_.nvEncUnregisterResource(encoder_session_, buffer.registered_resource);
      }
      if (buffer.bitstream_buffer) {
        nvenc_.nvEncDestroyBitstreamBuffer(encoder_session_, buffer.bitstream_buffer);
      }
    }
  }

  for (auto & buffer : frame_buffers_) {
    if (buffer.yuv_device) {
      CHECK_CUDA(cudaFree(buffer.yuv_device));
    }
  }
  frame_buffers_.clear();

  if (rgb_device_) {
    CHECK_CUDA(cudaFree(rgb_device_));
    rgb_device_ = nullptr;
  }

  if (encoder_session_) {
    nvenc_.nvEncDestroyEncoder(encoder_session_);
    encoder_session_ = nullptr;
  }

  if (stream_) {
    CHECK_CUDA(cudaStreamDestroy(stream_));
    stream_ = {};
  }

  if (nvenc_library_) {
    dlclose(nvenc_library_);
    nvenc_library_ = nullptr;
  }
}
#endif  // NVENC_AVAILABLE
}  // namespace accelerated_image_processor::compression
