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

#include "accelerated_image_processor_common/helper.hpp"
#include "accelerated_image_processor_decompression/video_decompressor.hpp"

#include <accelerated_image_processor_common/datatype.hpp>

#include <cuda.h>  // for driver API
#include <cuda_runtime.h>
#include <nppi_color_conversion.h>
#include <nppi_support_functions.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/packet.h>
#include <libavutil/buffer.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
}

namespace accelerated_image_processor::decompression
{
class FfmpegVideoDecompressor final : public VideoDecompressor
{
public:
  explicit FfmpegVideoDecompressor(common::ParameterMap dedicated_parameters = {})
  : VideoDecompressor(VideoBackend::FFMPEG, dedicated_parameters)
  {
    // Create a CUDA stream and shared it with NPP
    CHECK_CUDA(cudaStreamCreate(&stream_));
    {
      npp_stream_ctx_.hStream = stream_;
      CHECK_CUDA(cudaGetDevice(&npp_stream_ctx_.nCudaDeviceId));
      cudaDeviceProp dev_prop;
      CHECK_CUDA(cudaGetDeviceProperties(&dev_prop, npp_stream_ctx_.nCudaDeviceId));
      npp_stream_ctx_.nMultiProcessorCount = dev_prop.multiProcessorCount;
      npp_stream_ctx_.nMaxThreadsPerMultiProcessor = dev_prop.maxThreadsPerMultiProcessor;
      npp_stream_ctx_.nMaxThreadsPerBlock = dev_prop.maxThreadsPerBlock;
      npp_stream_ctx_.nSharedMemPerBlock = dev_prop.sharedMemPerBlock;
      CHECK_CUDA(cudaDeviceGetAttribute(
        &npp_stream_ctx_.nCudaDevAttrComputeCapabilityMajor, cudaDevAttrComputeCapabilityMajor,
        npp_stream_ctx_.nCudaDeviceId));
      CHECK_CUDA(cudaDeviceGetAttribute(
        &npp_stream_ctx_.nCudaDevAttrComputeCapabilityMinor, cudaDevAttrComputeCapabilityMinor,
        npp_stream_ctx_.nCudaDeviceId));
      CHECK_CUDA(cudaStreamGetFlags(npp_stream_ctx_.hStream, &npp_stream_ctx_.nStreamFlags));
    }
  }

  ~FfmpegVideoDecompressor()
  {
    cleanup_decoder();
    if (dst_bgr_dev_) {
      nppiFree(dst_bgr_dev_);
    }
  }

  /**
   * Override process function that takes one ffmpeg packet and calls the registered post process
   * for every decoded frame (NOTE: one ffmpeg packet may contain multiple frames)
   */
  std::optional<common::Image> process(const common::Image & image) override
  {
    auto processed_vec = this->process_packet(image);

    for (auto & processed : processed_vec) {
      this->postprocess(processed);
    }

    // For streaming cases, although one ffmpeg packet MAY contain multiple frames, it typically
    // contains one frame. For this reason, here returns the first element of the decoded results.
    if (processed_vec.empty()) {
      return std::nullopt;
    } else {
      return processed_vec[0];
    }
  }

private:
  struct DecoderInitResult
  {
    bool success;
    std::string error_msg;

    DecoderInitResult(bool is_success, std::string msg) : success(is_success), error_msg(msg) {}
  };

  void cleanup_decoder()
  {
    if (packet_) {
      av_packet_free(&packet_);
    }
    if (codec_ctx_) {
      avcodec_free_context(&codec_ctx_);
    }
    if (hw_device_ctx_) {
      av_buffer_unref(&hw_device_ctx_);
    }
  }

  std::string image_format_to_string(const common::ImageFormat & fmt) const
  {
    switch (fmt) {
      case common::ImageFormat::H264:
        return "h264";
      case common::ImageFormat::H265:
        return "hevc";
      case common::ImageFormat::AV1:
        return "av1";
      default:
        throw std::runtime_error("Unsuported format was detected");
    }
  }

  DecoderInitResult init_decoder(const common::ImageFormat & fmt)
  {
    // Allocate the HW device context (but not initialize it yet)
    //   This creates the structure but leaves the internal CUDA context pending
    // TODO(manato): expose device type as a parameter would beneficial especially for non-CUDA user
    hw_device_ctx_ = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_CUDA);
    if (!hw_device_ctx_) {
      return DecoderInitResult(false, "Failed to allocate CUDA HW device context");
    }

    // Inject the CUDA Stream
    //   safely modify the specific HW context settings nwo because initialization hasn't happened
    //   yet
    {
      // Extract CUcontext from cuda stream using CUDA driver API
      CUcontext cu_context;
      if (cuStreamGetCtx(reinterpret_cast<CUstream>(stream_), &cu_context) != CUDA_SUCCESS) {
        return DecoderInitResult(false, "Extracting CUcontext failed");
      }
      AVHWDeviceContext * av_hw_device_ctx =
        reinterpret_cast<AVHWDeviceContext *>(hw_device_ctx_->data);
      AVCUDADeviceContext * av_cuda_device_ctx =
        reinterpret_cast<AVCUDADeviceContext *>(av_hw_device_ctx->hwctx);
      av_cuda_device_ctx->stream = stream_;
      av_cuda_device_ctx->cuda_ctx = cu_context;
    }

    // Initialize the HW device context
    //   FFmpeg will now create the underlying CUContext (since we left cuda_ctx->cuda_ctx null)
    //   but will respect the stream we already assigned
    int err = av_hwdevice_ctx_init(hw_device_ctx_);
    if (err < 0) {
      char err_buf[128];
      av_strerror(err, err_buf, sizeof(err_buf));
      return DecoderInitResult(false, std::string("Failed to init HW device context: ") + err_buf);
    }

    // Find decoder
    std::string codec_name = image_format_to_string(fmt);
    const AVCodec * codec = avcodec_find_decoder_by_name(codec_name.c_str());
    if (!codec) {
      return DecoderInitResult(false, "Codec not found: " + codec_name);
    }

    // Allocate and Setup context
    codec_ctx_ = avcodec_alloc_context3(codec);
    codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
    codec_ctx_->get_format =
      []([[maybe_unused]] AVCodecContext * ctx, const enum AVPixelFormat * pix_fmts) {
        const enum AVPixelFormat * p;
        for (p = pix_fmts; *p != -1; p++) {  // We strictly look for the CUDA format
          if (*p == AV_PIX_FMT_CUDA) {
            return *p;
          }
        }
        return AV_PIX_FMT_NONE;
      };  // Set the callback to enforce CUDA format

    // Open Codec
    if (avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
      return DecoderInitResult(false, "Failed to open codec");
    }

    // Allocate packet wrapper
    packet_ = av_packet_alloc();
    if (!packet_) {
      return DecoderInitResult(false, "Failed to allcate AVPacket");
    }

    // Allocate region to store the decoded result
    decoded_frame_ = av_frame_alloc();
    if (!decoded_frame_) {
      return DecoderInitResult(false, "Failed to allcate AVFrame");
    }

    return DecoderInitResult(true, "");
  }

  /**
   * @brief core process implementation that handles input image (ffmpeg packat) and returns the
   * vector of Image
   *
   * Because one ffmpeg packet may include multiple frames, process_imple, which is the pure virtual
   * function that returns one image as a decoding result,  can not be applicable for this class.
   */
  std::vector<common::Image> process_packet(const common::Image & image)
  {
    // Initialize decoder for the first attempt
    if (!is_ready()) {
      if (auto res = init_decoder(image.format); !res.success) {
        std::cerr << "Failed to initialize decoder:  " << res.error_msg << std::endl;
        return std::vector<common::Image>();
      }
      log_once("Succeed to init_decoder");
    }

    // Pack received data into ffmpeg format
    packet_->data = const_cast<uint8_t *>(image.data.data());
    packet_->size = image.data.size();
    packet_->pts = image.pts.value();
    packet_->flags = image.flags.value();

    // Feed packet data to the decoder
    int ret = avcodec_send_packet(codec_ctx_, packet_);
    if (ret < 0) {
      std::cerr << "Packet send error" << std::endl;
      return std::vector<common::Image>();
    }

    std::vector<common::Image> processed_vec;
    while (ret >= 0) {
      // Receive decoded result
      // NOTE: avcodec_receive_frame automatically calls av_frame_unref(frame_)
      // internally before writing new data, so it is safe to pass a used frame
      ret = avcodec_receive_frame(codec_ctx_, decoded_frame_);
      if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        av_frame_unref(decoded_frame_);
        break;
      } else if (ret < 0) {
        std::cerr << "Error during decoding" << std::endl;
        // Return the results so far
        return processed_vec;
      }

      // color space conversion
      common::Image output_image;
      {
        output_image.frame_id = image.frame_id;
        output_image.timestamp = image.timestamp;
        output_image.height = image.height;
        output_image.width = image.width;
        output_image.format = common::ImageFormat::RAW;
        output_image.step = image.width * 3;
        output_image.encoding = common::ImageEncoding::BGR;
        output_image.data = yuv_to_bgr(decoded_frame_);
      }
      processed_vec.push_back(std::move(output_image));

      // Release references explicitly
      // Although receive_frame unrefs at the start, we manually unref here
      // to return the GPU surface to the pool IMMEDIATELY.
      // If we wait until the next callback, we might starve teh decoder's surface pool
      av_frame_unref(decoded_frame_);
    }

    return processed_vec;
  }

  std::vector<uint8_t> yuv_to_bgr(AVFrame * frame)
  {
    size_t width_in_byte = frame->width * 3;  // BGR

    // Re-allocate presistent GPU BGR buffer if resolution changes
    if (dst_width_ != frame->width || dst_height_ != frame->height) {
      if (dst_bgr_dev_) {
        nppiFree(dst_bgr_dev_);
      }

      dst_bgr_dev_ = nppiMalloc_8u_C3(frame->width, frame->height, &dst_pitch_);

      dst_width_ = frame->width;
      dst_height_ = frame->height;
    }

    // Color conversion (NV12/YUV444 -> BGR) using NPP
    NppiSize roi = {frame->width, frame->height};
    int src_step = frame->linesize[0];

    // Identify the decoded pixel format
    AVPixelFormat decoded_format = AV_PIX_FMT_NONE;
    if (frame->hw_frames_ctx) {
      auto * ctx = reinterpret_cast<AVHWFramesContext *>(frame->hw_frames_ctx->data);
      decoded_format = ctx->sw_format;
    } else {
      decoded_format = static_cast<AVPixelFormat>(frame->format);
    }

    // Check the underlying software format to decide the conversion API
    // (The output format is automatically determined (will be the same as) encoded data)
    switch (decoded_format) {
      case AV_PIX_FMT_NV12: {
        log_once("AV_PIX_FMT_NV12 format detected");
        // NV12 holds 2 planes
        const Npp8u * d_src[2] = {frame->data[0], frame->data[1]};
        CHECK_NPP(nppiNV12ToBGR_8u_P2C3R_Ctx(
          d_src, src_step, dst_bgr_dev_, dst_pitch_, roi, npp_stream_ctx_));
        break;
      }
      case AV_PIX_FMT_YUV444P: {
        log_once("AV_PIX_FMT_YUV444P format detected");
        // YUV444 holds 3 planes
        const Npp8u * d_src[3] = {frame->data[0], frame->data[1], frame->data[2]};
        CHECK_NPP(nppiYUVToBGR_8u_P3C3R_Ctx(
          d_src, src_step, dst_bgr_dev_, dst_pitch_, roi, npp_stream_ctx_));
        break;
      }
      default: {
        std::cerr << "Unsupported format: " << av_get_pix_fmt_name(decoded_format) << std::endl;
      }
    }

    // Download the data
    std::vector<uint8_t> bgr_host;
    bgr_host.resize(width_in_byte * dst_height_);
    CHECK_CUDA(cudaMemcpy2DAsync(
      bgr_host.data(), width_in_byte, dst_bgr_dev_, dst_pitch_, width_in_byte, dst_height_,
      cudaMemcpyDeviceToHost, stream_));

    CHECK_CUDA(cudaStreamSynchronize(stream_));
    return bgr_host;
  }

  /**
   * @brief Utility function that print message only once
   */
  inline void log_once(const std::string & msg)
  {
    static std::once_flag flag;
    std::call_once(flag, [&]() { std::cout << msg << std::endl; });
  }

  /**
   * @brief dummy override for the pure virtual function. This will not be used in this class
   */
  common::Image process_impl(const common::Image & image) override { return image; }

  /**
   * @brief Checks if the decompressor is ready for processing.
   *
   * @return true if the codec context, hardware device context, and decoded frame
   *         have been successfully initialized; false otherwise.
   */
  bool is_ready() const override { return (codec_ctx_) && (hw_device_ctx_) && (decoded_frame_); }

  // FFmpeg stuff
  AVPacket * packet_{nullptr};
  AVCodecContext * codec_ctx_{nullptr};
  AVBufferRef * hw_device_ctx_{nullptr};
  AVFrame * decoded_frame_{nullptr};

  // CUDA resources to store BGR converted image
  uint8_t * dst_bgr_dev_{nullptr};
  int dst_pitch_{0};
  int dst_width_{0};
  int dst_height_{0};
  cudaStream_t stream_;
  NppStreamContext npp_stream_ctx_;
};

std::unique_ptr<VideoDecompressor> make_ffmpeg_video_decompressor()
{
  return std::make_unique<FfmpegVideoDecompressor>();
}

}  // namespace accelerated_image_processor::decompression
