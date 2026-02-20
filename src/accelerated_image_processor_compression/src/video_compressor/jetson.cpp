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

#include "jetson.hpp"

#include "accelerated_image_processor_compression/video_compressor.hpp"
#include "jetson_error_helper.hpp"
#include "jetson_precise_timestamp_map.hpp"

#include <accelerated_image_processor_common/helper.hpp>

#include <endian.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>

// Header that defines ffmpeg flags
extern "C" {
#include <libavcodec/avcodec.h>
}

namespace
{
constexpr bool is_big_endian = (__BYTE_ORDER__ == __BIG_ENDIAN);
}  // namespace

namespace accelerated_image_processor::compression
{

EncResult JetsonVideoCompressor::collect_params(EncoderParameter & params)
{
  params.buffer_length = this->parameter_value<int>("buffer_length");
  params.compression_type = string_to_enum<VideoCompressionType>(
    this->parameter_value<std::string>("compression_type"), video_compression_type_map);

  params.idr_interval = this->parameter_value<int>("idr_frame_interval");
  params.i_frame_interval = this->parameter_value<int>("i_frame_interval");
  params.frame_rate_numerator = this->parameter_value<int>("frame_rate_numerator");
  params.frame_rate_denominator = this->parameter_value<int>("frame_rate_denominator");
  params.hw_preset_type = string_to_enum<v4l2_enc_hw_preset_type>(
    this->parameter_value<std::string>("hw_preset_type"), hardware_preset_map);
  params.use_max_performance_mode = this->parameter_value<bool>("use_max_performance");
  params.target_bits_per_pixel = this->parameter_value<double>("target_bits_per_pixel");

  return EncResult{EncStatus{true, ""}};
}

EncResult JetsonVideoCompressor::init_encoder(const common::Image & image)
{
  // gather parameters
  if (auto res = collect_params(encoder_params_); !res.ok) {
    return EncResult(record_error("Failed to correct parameters (" + res.status.message + ")"));
  }

  if (auto res = this->collect_codec_params_impl(encoder_params_); !res.ok) {
    return EncResult(
      record_error("Failed to correct codec dedicated parameters (" + res.status.message + ")"));
  }

  // Confirm the given combination of parameters is valid
  if (auto [is_valid, msg] = validate_compression_type_compatibility(); !is_valid) {
    return EncResult(record_error("Invalid parameters (" + msg + ")"));
  }

  output_plane_fds_.assign(encoder_params_.buffer_length, -1);
  output_nvsurface_.assign(encoder_params_.buffer_length, nullptr);
  timestamp_map_ = std::make_unique<TimestampMap>(encoder_params_.buffer_length);

  // Configure encoder output (codec individual)
  {
    if (auto res =
          this->set_capture_plane_format_impl(image.width, image.height, image.step * image.height);
        !res.ok)
      return EncResult(
        record_error("Failed to set capture plane format (" + res.status.message + ")"));
  }

  // Configure encoder input
  {
    auto pixel_format = pixel_format_map.at(encoder_params_.compression_type);
    CHECK_NVENC(
      encoder_->setOutputPlaneFormat(pixel_format, image.width, image.height),
      "Failed to set output plane format");
  }

  // Do codec specific configuration
  //   almost all them are specified to be executed after setting ouput/capture plane format
  //   and before requesting any plane buffers
  {
    if (auto res = this->init_codec_impl(); !res.ok) {
      return EncResult(
        record_error("Codec specific configuration failed (" + res.status.message + ")"));
    }
  }

  // Common configurations
  {
    if (encoder_params_.compression_type == VideoCompressionType::LOSSLESS) {
      CHECK_NVENC(encoder_->setLossless(true), "Could not set lossless encoding");
    } else {
      // Enable variable rate control (VRC)
      CHECK_NVENC(
        encoder_->setRateControlMode(V4L2_MPEG_VIDEO_BITRATE_MODE_VBR),
        "Failed to set variable rate control mode");

      // compute the target bit rate from input streaming rate
      double frame_rate = static_cast<double>(encoder_params_.frame_rate_numerator) /
                          static_cast<double>(encoder_params_.frame_rate_denominator);
      auto target_bit_rate =
        image.height * image.width * frame_rate * encoder_params_.target_bits_per_pixel;
      auto peak_bit_rate = 1.2 * target_bit_rate;  // set 1.2x of average bitrate as peak bitrate
      CHECK_NVENC(encoder_->setBitrate(target_bit_rate), "Failed to set bit rate");
      CHECK_NVENC(encoder_->setPeakBitrate(peak_bit_rate), "Failed to set peak bit rate");
    }

    // Set IDR (Instantaneous Decoding Refresh) frame interval
    // The IDR frame is a special format of I frame that ensures later P (and B) frames never refer
    // the I frames before this frame. Decoders can restart from this frame in cases of seek or
    // error.
    CHECK_NVENC(
      encoder_->setIDRInterval(encoder_params_.idr_interval), "Failed to set encoder IDR interval");

    // Set I frame interval
    // I frame is self-decodable frame, which can be decoded without referring other frames
    CHECK_NVENC(
      encoder_->setIFrameInterval(encoder_params_.i_frame_interval),
      "Failed to set I Frame interval");

    // Set frame rate
    // rate is specified in [numerator (second), denominator (frames)] format
    CHECK_NVENC(
      encoder_->setFrameRate(
        encoder_params_.frame_rate_numerator, encoder_params_.frame_rate_denominator),
      "Failed to set frame rate");

    CHECK_NVENC(
      encoder_->setTemporalTradeoff(
        V4L2_ENC_TEMPORAL_TRADEOFF_LEVEL_DROPNONE),  // encode all frames
      "Failed to set temporal trade off level to DROPNONE");

    CHECK_NVENC(
      encoder_->setHWPresetType(encoder_params_.hw_preset_type),
      "Failed to set encoder hardware preset type");

    // Video Usability Information (VUI) and extended color format are required to embed source
    // image information properly
    CHECK_NVENC(
      encoder_->setInsertVuiEnabled(true), "Failed to set insert Video Usability information");
    CHECK_NVENC(encoder_->setExtendedColorFormat(true), "Failed to set extended color format");

    CHECK_NVENC(encoder_->setAlliFramesEncode(false), "Failed to set number of all I frame");

    // Disable B-Frame for streaming compression
    CHECK_NVENC(encoder_->setNumBFrames(0), "Failed to set number of B-Frames");

    CHECK_NVENC(
      encoder_->setMaxPerfMode(static_cast<int>(encoder_params_.use_max_performance_mode)),
      "Error while setting encoder to max performance");
  }

  // Configure output plane (encoder input) so that it allows direct memory access buffer
  {
    if (auto res = setup_output_plane(image.height, image.width); !res.ok) {
      return EncResult(
        record_error("Failed to setup output DMA buffer (" + res.status.message + ")"));
    }
  }

  // Export and Map the capture plane buffers so that we can write encoded
  // bistream data into the buffers
  // NOTE: "capture_plane" represents "OUTPUT" of the encoder (the place
  // where the application receives encoded images)
  {
    CHECK_NVENC(
      encoder_->capture_plane.setupPlane(
        V4L2_MEMORY_MMAP, encoder_params_.buffer_length, true, false),
      "Could not setup capture plane");
  }

  // Boot encoder
  {
    // Subscribe for End Of Stream event
    CHECK_NVENC(encoder_->subscribeEvent(V4L2_EVENT_EOS, 0, 0), "Failed to subscribe EOS event");

    // set encoder output plane (encoder input) STREAMON
    CHECK_NVENC(encoder_->output_plane.setStreamStatus(true), "Error in output plane streamon");

    // set encoder capture plane (encoder output) STREAMON
    CHECK_NVENC(encoder_->capture_plane.setStreamStatus(true), "Error in capture plane streamon");
  }

  // Start to run encoder callback thread that receives encoded results
  {
    // Set encoder capture plane dq thread callback for blocking io mode
    encoder_->capture_plane.setDQThreadCallback(encoder_capture_plane_dq_callback);

    // startDQThread starts a thread internally which calls the encoder_capture_plane_dq_callback
    // whenever a buffer is dequeued on the plane
    dq_callback_args_ =
      std::make_unique<DqCallbackArgs>(image.frame_id, image.width, image.height, this);

    CHECK_NVENC(
      encoder_->capture_plane.startDQThread(dq_callback_args_.get()),
      "Failed to satart encoder callback thread");
  }

  // Enqueue all the empty capture_plane buffers
  for (uint32_t i = 0; i < encoder_->capture_plane.getNumBuffers(); i++) {
    struct v4l2_buffer v4l2_buf;
    struct v4l2_plane planes[MAX_PLANES];

    std::memset(&v4l2_buf, 0, sizeof(v4l2_buf));
    std::memset(&planes, 0, sizeof(v4l2_plane));

    v4l2_buf.index = i;
    v4l2_buf.m.planes = planes;

    CHECK_NVENC(
      encoder_->capture_plane.qBuffer(v4l2_buf, NULL),
      "Error while queueing buffer at capture plane");
  }

  // Enqueue all the empty output_plane buffers
  for (uint32_t i = 0; i < encoder_->output_plane.getNumBuffers(); i++) {
    struct v4l2_buffer v4l2_buf;
    struct v4l2_plane planes[MAX_PLANES];

    std::memset(&v4l2_buf, 0, sizeof(v4l2_buf));
    std::memset(planes, 0, sizeof(planes));

    v4l2_buf.index = i;
    v4l2_buf.m.planes = planes;
    v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    v4l2_buf.memory = V4L2_MEMORY_DMABUF;
    CHECK_NVENC(
      encoder_->output_plane.mapOutputBuffers(v4l2_buf, output_plane_fds_[i]),
      "Error while mapping buffer at output plane");

    // zero clear output_plane memory at once
    CHECK_NVENC(NvBufSurfaceMemSet(output_nvsurface_[i], 0, -1, 0), "Failed to NvBufSurfaceMemSet");

    NvBuffer * buffer = encoder_->output_plane.getNthBuffer(i);
    // Sync the hardware memory cache for the device
    for (uint32_t j = 0; j < buffer->n_planes; j++) {
      // zero clear and set bytesused member to non zero value
      auto & plane = buffer->planes[j];
      uint32_t max_size = plane.fmt.sizeimage;

      plane.bytesused = max_size;

      v4l2_buf.m.planes[j].bytesused = max_size;

      CHECK_NVENC(
        NvBufSurfaceSyncForDevice(output_nvsurface_[i], 0, -1),
        "Error while NvBufSurfaceSyncFor Device at output plane for V4L2_MEMORY_DMABUF");
    }

    CHECK_NVENC(
      encoder_->output_plane.qBuffer(v4l2_buf, NULL),
      "Error while queueing buffer at output plane");
  }

  // Now, ready to process
  state_ = State::READY;
  return EncResult::success();
}

EncResult JetsonVideoCompressor::setup_output_plane(const int & height, const int & width)
{
  CHECK_NVENC(
    encoder_->output_plane.reqbufs(V4L2_MEMORY_DMABUF, encoder_params_.buffer_length),
    "Failed to request buffer for output plane as V4L2_MEMORY_DMABUF");

  for (uint32_t i = 0; i < encoder_->output_plane.getNumBuffers(); i++) {
    NvBufSurfaceCreateParams nvbuf_surface_create_params = {};
    nvbuf_surface_create_params.width = width;
    nvbuf_surface_create_params.height = height;
    nvbuf_surface_create_params.layout = NVBUF_LAYOUT_PITCH;
    nvbuf_surface_create_params.colorFormat =
      nvbuf_color_format_map.at(encoder_params_.compression_type);
    nvbuf_surface_create_params.memType = NVBUF_MEM_SURFACE_ARRAY;

    /* Create output plane fd for DMABUF io-mode */
    CHECK_NVENC(
      NvBufSurfaceCreate(&output_nvsurface_[i], 1, &nvbuf_surface_create_params),
      "Failed to create NvBufSurface");
    output_plane_fds_[i] = output_nvsurface_[i]->surfaceList[0].bufferDesc;
    output_nvsurface_[i]->numFilled = 1;
  }

  return EncResult::success();
}

common::Image JetsonVideoCompressor::process_impl(const common::Image & image)
{
  if (state_ != State::READY) {
    if (!init_encoder(image).ok) {
      throw std::runtime_error("Encoder initialization failed: " + last_error_);
    }
  }

  // queue video frame the output plane buffer
  struct v4l2_buffer v4l2_buf;
  struct v4l2_plane planes[MAX_PLANES];
  NvBuffer * buffer;

  std::memset(&v4l2_buf, 0, sizeof(v4l2_buf));
  std::memset(planes, 0, sizeof(planes));
  v4l2_buf.m.planes = planes;

  // Dequeue buffer from encoder output plane
  if (auto ret = encoder_->output_plane.dqBuffer(v4l2_buf, &buffer, NULL, 10); ret < 0) {
    std::cerr << "ERROR while DQing buffer at output plane" << std::endl;
    return common::Image();
  }

  // Wrap input data in VPIImage so that the data can be handled by VPI transparently
  {
    VPIImageData input_data_params = {};
    input_data_params.bufferType = VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR;
    if (image.encoding == common::ImageEncoding::RGB) {
      input_data_params.buffer.pitch.format = VPI_IMAGE_FORMAT_RGB8;
    } else if (image.encoding == common::ImageEncoding::BGR) {
      input_data_params.buffer.pitch.format = VPI_IMAGE_FORMAT_BGR8;
    } else {
      std::cerr << "Unsupported input encoding detected" << std::endl;
      return common::Image();
    }
    input_data_params.buffer.pitch.numPlanes = 1;
    input_data_params.buffer.pitch.planes[0].pixelType = VPI_PIXEL_TYPE_3U8;
    input_data_params.buffer.pitch.planes[0].width = image.width;
    input_data_params.buffer.pitch.planes[0].height = image.height;
    input_data_params.buffer.pitch.planes[0].pitchBytes = image.step;
    input_data_params.buffer.pitch.planes[0].data = const_cast<uint8_t *>(image.data.data());

    VPIImageWrapperParams input_wrapper_params = {};
    CHECK_VPI(vpiInitImageWrapperParams(&input_wrapper_params));
    input_wrapper_params.colorSpec =
      VPI_COLOR_SPEC_DEFAULT;  // Informs that the color spec is to be inferred.
                               // input_wrapper_params.colorSpec = VPI_COLOR_SPEC_sRGB;
    if (!input_rgb_dev_) {
      CHECK_VPI(
        vpiImageCreateWrapper(&input_data_params, &input_wrapper_params, 0, &input_rgb_dev_));
    } else {
      CHECK_VPI(vpiImageSetWrapper(input_rgb_dev_, &input_data_params));
    }
  }

  // Wrap the memory region held by NvBuffer so that VPI can write the
  // color conversion result to it directly
  {
    VPIImageData data_params = {};
    data_params.bufferType = VPI_IMAGE_BUFFER_NVBUFFER;
    data_params.buffer.fd = output_plane_fds_[v4l2_buf.index];

    VPIImageWrapperParams wrapper_params = {};
    CHECK_VPI(vpiInitImageWrapperParams(&wrapper_params));
    wrapper_params.colorSpec =
      VPI_COLOR_SPEC_DEFAULT;  // Informs that the color spec is to be inferred.

    uint64_t wrapper_flag = 0;  // The backend selection happens during the algorithm submission
    if (!output_yuv_dev_) {
      // Create an image object by wrapping an existing memory block.
      CHECK_VPI(
        vpiImageCreateWrapper(&data_params, &wrapper_params, wrapper_flag, &output_yuv_dev_));
    } else {
      // Redefines the wrapped memory in an existing VPIImage wrapper.
      CHECK_VPI(vpiImageSetWrapper(output_yuv_dev_, &data_params));
    }
  }

  // fill encoder input memory region with the color converted image data
  fill_encoder_input_async();

  // Copy timestamp from source image
  {
    // NOTE: Since nanosecond order timestamp resolution, such as provided by ROS timestamp, will be
    // lost in v4l2_buf.timestamp (microsecond order), actual timestamp is derived to the output
    // result via timestamp_map_
    v4l2_buf.flags |= V4L2_BUF_FLAG_TIMESTAMP_COPY;
    v4l2_buf.timestamp.tv_sec = image.timestamp / 1'000'000'000ULL;
    v4l2_buf.timestamp.tv_usec = (image.timestamp % 1'000'000'000ULL) / 1'000ULL;
    timestamp_map_->set(v4l2_buf.index, TimestampMap::PreciseTimestamp(image.timestamp));
  }

  // Since filling frame data to the buffer goes asynchronously,
  // wait its completion here before conduct qBuffer (i.e., encoding)
  {
    CHECK_CUDA(cudaStreamSynchronize(stream_));
    CHECK_VPI(vpiStreamSync(vpi_stream_));
  }

  // Sync the hardware memory cache for the device
  for (uint32_t j = 0; j < buffer->n_planes; j++) {
    NvBufSurface * nvbuf_surf = 0;
    if (auto ret = NvBufSurfaceFromFd(buffer->planes[j].fd, reinterpret_cast<void **>(&nvbuf_surf));
        ret < 0) {
      std::cerr << "Error while NvBufSurfaceFromFd" << std::endl;
      return common::Image();
    }
    if (auto ret = NvBufSurfaceSyncForDevice(nvbuf_surf, 0, j); ret < 0) {
      std::cerr << "Error while NvBuSurfaceSyncForDevice at output plane for V4L2_MEMORY_MMAP"
                << std::endl;
      return common::Image();
    }

    buffer->planes[j].bytesused = buffer->planes[j].fmt.stride * buffer->planes[j].fmt.height;
    v4l2_buf.m.planes[j].bytesused = buffer->planes[j].bytesused;
  }

  // feed input data to the encoder
  {
    if (auto ret = encoder_->output_plane.qBuffer(v4l2_buf, NULL); ret < 0) {
      std::cerr << "Error while queueing buffer at output plane" << std::endl;
      return common::Image();
    }
  }

  // Since the compression result will be acquired on the other thread,
  // just return input as a valid data
  return image;
}

void JetsonVideoCompressor::fill_encoder_input_async(void)
{
  uint64_t backend = VPI_BACKEND_CUDA;
  VPIConvertImageFormatParams cvt_params;
  vpiInitConvertImageFormatParams(&cvt_params);
  cvt_params.policy = VPI_CONVERSION_CAST;
  cvt_params.flags = VPI_PRECISE;

  CHECK_VPI(vpiSubmitConvertImageFormat(
    vpi_stream_, backend, input_rgb_dev_, output_yuv_dev_, &cvt_params));
}

bool JetsonVideoCompressor::encoder_capture_plane_dq_callback(
  struct v4l2_buffer * v4l2_buf, NvBuffer * buffer, [[maybe_unused]] NvBuffer * shared_buffer,
  void * arg)
{
  // Argument provided via startDQThread() in void* form
  DqCallbackArgs * callback_args = reinterpret_cast<DqCallbackArgs *>(arg);
  auto * compressor_object = callback_args->obj;
  auto * encoder = compressor_object->encoder();

  if (v4l2_buf == nullptr) {
    encoder->abort();
    std::cerr << "Error while dequeuing buffer from capture plane" << std::endl;
    return false;
  }

  // Execute preprocess for the encoded payload (some codec requires dedicated handling)
  auto payload_info =
    compressor_object->payload_preprocess_impl(callback_args, v4l2_buf->bytesused, buffer);

  // Since this function will also be called during initialization, which dummy frames are fed,
  // skip such dummy data
  {
    auto & initial_frame_count = compressor_object->initial_frame_count();
    const auto buffer_length = compressor_object->encoder_params().buffer_length;
    if (initial_frame_count < buffer_length) {
      // Do nothing for the dummy data. Just return (queue) buffer to the capture plane so that
      // successive actual frames arrive
      initial_frame_count++;
      CHECK_ERROR(
        encoder->capture_plane.qBuffer(*v4l2_buf, NULL) < 0,
        "Failed to queuing buffer to the capture plane");
      return true;
    }
  }

  // Get encode metadata
  v4l2_ctrl_videoenc_outputbuf_metadata enc_metadata;
  encoder->getMetadata(v4l2_buf->index, enc_metadata);

  // Create result data
  common::Image processed;
  {
    auto & timestamp_map = compressor_object->timestamp_map();
    int64_t stamp_in_nanosecond = 0;
    TimestampMap::PreciseTimestamp ps;
    if (!timestamp_map->get(v4l2_buf->index, ps)) {
      // fail to fetch precise timestamp. Fallback to use v4l2 buffer timestamp
      stamp_in_nanosecond = static_cast<int64_t>(v4l2_buf->timestamp.tv_sec) * 1e9 +
                            static_cast<int64_t>(v4l2_buf->timestamp.tv_usec) * 1e3;
    } else {
      stamp_in_nanosecond = static_cast<int64_t>(ps);
    }

    processed.frame_id = callback_args->frame_id;
    processed.timestamp = stamp_in_nanosecond;
    processed.height = callback_args->input_height;
    processed.width = callback_args->input_width;
    processed.format = supported_codec_format_map.at(compressor_object->codec());
    processed.pts = stamp_in_nanosecond / 1e3;  // [us]
    processed.flags = enc_metadata.KeyFrame ? AV_PKT_FLAG_KEY : 0;
    processed.is_bigendian = is_big_endian;

    compressor_object->payload_copy_impl(
      enc_metadata.KeyFrame, payload_info, callback_args, processed.data);
  }

  // Now, v4l2_buffer can be queued again
  CHECK_ERROR(
    encoder->capture_plane.qBuffer(*v4l2_buf, NULL) < 0,
    "Failed to Queuing buffer to capture plane");

  // call postprocess
  compressor_object->post_process(processed);

  return true;
}

}  // namespace accelerated_image_processor::compression
