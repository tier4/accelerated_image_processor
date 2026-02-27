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

#include <gtest/gtest.h>
#include <libavutil/error.h>

#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

#include <accelerated_image_processor_common/datatype.hpp>

namespace accelerated_image_processor::decompression
{

/**
 * @brief Test helper that produces a stream of encoded frames using FFmpeg.
 *
 * The class is parameterised by the codec name (`h264`, `h265`, `av1`) and
 * the number of frames to generate.  It exposes a `next()` method that
 * returns a `common::Image` containing the *encoded* packet.  The packet
 * can then be fed to a `VideoDecompressor` instance.
 *
 * NOTE: This provider emulates the following command to generate encoded payload
 *   ````
 *   ffmpeg -f lavfi -i testsrc=size=<WIDTH>x<HEIGHT>:rate=<FPS> \
 *     -vframes <NUM_FRAMES> -c:v <CodecName::value> -b:v <BITRATE> -g 10 \
 *     -pix_fmt yuv420p -f <CodecName::format> -
 *   ```
 *`
 *  Options' description
 *  -f lavfi -i testsrc=…=
 *      Creates a synthetic video source (the same filter graph the node builds).
 *  -vframes 20
 *     Stop after 20 frames – matches NUM_FRAMES.
 *  -c:v libx264
 *      Use the same encoder (libx264 for H.264).
 *      Replace with libx265 or libaom-av1 for the other codecs.
 *  -b:v 400k
 *      Target bitrate (400 kbps)
 *  -g 10
 *      GOP size (10).
 *  -pix_fmt yuv420p
 *      Same pixel format as the node.
 *  -f h264 / h265 / ivf
 *      Output format – raw bitstream (no container).
 *  output.<ext>
 *      Destination file (or - to pipe to stdout).
 */
template <typename CodecName>
class FfmpegTestDataProvider
{
public:
  static constexpr int NUM_FRAMES = 20;
  static constexpr int WIDTH = 2880;
  static constexpr int HEIGHT = 1860;
  static constexpr int FPS = 30;
  static constexpr int BITRATE = 400000;  // 400 kbps

  explicit FfmpegTestDataProvider()
  {
    // Embed frame count into the frame contents
    // NOTE: Large counter shown on the right middle depicts second of the video
    // NOTE: To embed frame count into the pixel value (so that we can confirm the count is as
    // expected programatically),
    //   `std::string filter_descr = "geq=lum='mod(N*10, 255)',format=pix_fmts=yuv420p";`
    // is another option. This increases pixel brightness by 10 (back to 0 if the frame number
    // reaches 255) frame by frame

    std::string filter_descr =
      "drawtext=text='%{n}':fontsize=150:fontcolor=white:x=100:y=100,format=pix_fmts=yuv420p";

    AVFilterGraph * graph = avfilter_graph_alloc();
    AVFilterContext * src_ctx = nullptr;
    AVFilterContext * sink_ctx = nullptr;

    const AVFilter * src = avfilter_get_by_name("testsrc");
    const AVFilter * sink = avfilter_get_by_name("buffersink");
    AVFilterInOut * inputs = avfilter_inout_alloc();
    AVFilterInOut * outputs = avfilter_inout_alloc();

    av_opt_set_int(src_ctx, "sample_aspect_ratio", 0, 0);
    av_opt_set_int(sink_ctx, "sample_aspect_ratio", 0, 0);

    // avfilter_graph_create_filter(&src_ctx, src, "src", filter_descr.c_str(), nullptr, graph);
    std::string src_filter_descr = "size=" + std::to_string(WIDTH) + "x" + std::to_string(HEIGHT) +
                                   ":rate=" + std::to_string(FPS);
    avfilter_graph_create_filter(&src_ctx, src, "src", src_filter_descr.c_str(), nullptr, graph);
    avfilter_graph_create_filter(&sink_ctx, sink, "sink", nullptr, nullptr, graph);

    outputs->name = av_strdup("in");
    outputs->filter_ctx = src_ctx;
    outputs->pad_idx = 0;
    outputs->next = nullptr;

    inputs->name = av_strdup("out");
    inputs->filter_ctx = sink_ctx;
    inputs->pad_idx = 0;
    inputs->next = nullptr;

    auto free_allocated_resources = [&graph, &inputs, &outputs]() {
      if (graph) {
        avfilter_graph_free(&graph);
      }
      if (inputs) {
        avfilter_inout_free(&inputs);
      }
      if (outputs) {
        avfilter_inout_free(&outputs);
      }
    };

    // av_log_set_level(AV_LOG_DEBUG);

    if (avfilter_graph_parse_ptr(graph, filter_descr.c_str(), &inputs, &outputs, nullptr) < 0) {
      free_allocated_resources();
      throw std::runtime_error("Failed to parse filter graph");
    }
    if (avfilter_graph_config(graph, nullptr) < 0) {
      free_allocated_resources();
      throw std::runtime_error("Failed to configure filter graph");
    }

    avfilter_inout_free(&inputs);
    avfilter_inout_free(&outputs);

    // 2. Find encoder
    const AVCodec * codec = avcodec_find_encoder_by_name(CodecName::value);
    if (!codec) {
      free_allocated_resources();
      throw std::runtime_error("Codec not found");
    }
    encoder_ctx_ = avcodec_alloc_context3(codec);
    encoder_ctx_->width = WIDTH;
    encoder_ctx_->height = HEIGHT;
    encoder_ctx_->time_base = AVRational{1, FPS};
    encoder_ctx_->framerate = AVRational{FPS, 1};
    encoder_ctx_->gop_size = 10;
    encoder_ctx_->max_b_frames = 0;
    encoder_ctx_->pix_fmt = AV_PIX_FMT_YUV420P;
    encoder_ctx_->bit_rate = BITRATE;

    // Enable encoder acceleration options to reduce test duration
    if (std::string(CodecName::value) == "libx264" || std::string(CodecName::value) == "libx265") {
      av_opt_set(encoder_ctx_->priv_data, "tune", "zerolatency", 0);
      av_opt_set(encoder_ctx_->priv_data, "preset", "ultrafast", 0);
    } else if (std::string(CodecName::value) == "libaom-av1") {
      // accelerated options for AV1
      // Without these, the test cases for AV1 takes over 2min, which causes colcon test timeout
      // cpu-used: It can specify 0--8. larger value lower compression (8 is fastest)
      av_opt_set(encoder_ctx_->priv_data, "cpu-used", "8", 0);
      // usage: set `realtime` to minimize delay and processing time
      av_opt_set(encoder_ctx_->priv_data, "usage", "realtime", 0);
    }

    if (avcodec_open2(encoder_ctx_, codec, nullptr) < 0) {
      free_allocated_resources();
      if (encoder_ctx_) {
        avcodec_free_context(&encoder_ctx_);
      }
      throw std::runtime_error("Could not open encoder");
    }

    // 3. Prepare packet buffer
    pkt_ = av_packet_alloc();
    frame_ = av_frame_alloc();
    frame_->format = encoder_ctx_->pix_fmt;
    frame_->width = encoder_ctx_->width;
    frame_->height = encoder_ctx_->height;
    av_frame_get_buffer(frame_, 32);

    // 4. Store graph for later use
    graph_ = graph;
    src_ctx_ = src_ctx;
    sink_ctx_ = sink_ctx;
  }

  ~FfmpegTestDataProvider()
  {
    av_packet_free(&pkt_);
    av_frame_free(&frame_);
    avcodec_free_context(&encoder_ctx_);
    avfilter_graph_free(&graph_);
  }

  /**
   * @brief Return the next encoded frame as a `common::Image`.
   *
   * The returned image contains the raw packet data in `data`.  The
   * `format` field is set to `JPEG` (or whatever the codec produces)
   * so that the decompressor can recognise it.
   */
  std::optional<common::Image> next()
  {
    while (true) {
      // Try to get encoded packet from encoder
      int ret = avcodec_receive_packet(encoder_ctx_, pkt_);

      if (ret == 0) {
        // Packet acquired successfully. Break the loop and return it
        break;
      } else if (ret == AVERROR(EAGAIN)) {
        // Encoder requires some frames to output actual encoded data. This error indicates the
        // encoder still needs more frame to start output encoded payload

        // If we already sent all frames to be tested,
        // send empty frame (nullptr) to notify encoder to finish the execution (flush)
        if (frame_index_ >= NUM_FRAMES) {
          avcodec_send_frame(encoder_ctx_, nullptr);
          continue;  // Try to receive packet again
        }

        // draw a new frame from filter graph
        if (av_buffersink_get_frame(sink_ctx_, frame_) < 0) {
          return std::nullopt;  // Exit if a new frame is unable to draw
        }

        // Set pts
        frame_->pts = frame_index_;

        // Send the frame to the encoder
        if (avcodec_send_frame(encoder_ctx_, frame_) < 0) {
          av_frame_unref(frame_);  // release the frame if it fails to send
          return std::nullopt;
        }

        // Once the sending is finished, release the frame to prepare next try
        av_frame_unref(frame_);

        // Go next frame
        frame_index_++;
      } else if (ret == AVERROR_EOF) {
        // Encode finished successfully
        return std::nullopt;
      } else {
        // Unexpected behavior
        throw std::runtime_error("Unexpected state found");
      }
    }

    // Build the Image
    common::Image img;
    img.frame_id = "ffmpeg_test";
    img.timestamp = 123456789 + (frame_index_ - 1) * 1000000 / FPS;  // 1s per frame
    img.width = WIDTH;
    img.height = HEIGHT;
    img.step = 0;                    // not relevant for compressed data
    img.format = CodecName::format;  // placeholder, adjust if needed
    img.pts = frame_index_ - 1;
    img.flags = (pkt_->flags & AV_PKT_FLAG_KEY) != 0;
    img.data.resize(pkt_->size);
    std::memcpy(img.data.data(), pkt_->data, pkt_->size);

    // Clean up packet for next use
    av_packet_unref(pkt_);

    return img;
  }

private:
  AVFilterGraph * graph_{nullptr};
  AVFilterContext * src_ctx_{nullptr};
  AVFilterContext * sink_ctx_{nullptr};

  AVCodecContext * encoder_ctx_{nullptr};
  AVFrame * frame_{nullptr};
  AVPacket * pkt_{nullptr};

  int frame_index_{0};
};

/**
 * Helper type aliases for the three codecs.
 */
struct H264Provider
{
  static constexpr const char * value = "libx264";
  static constexpr const common::ImageFormat format = common::ImageFormat::H264;
};
struct H265Provider
{
  static constexpr const char * value = "libx265";
  static constexpr const common::ImageFormat format = common::ImageFormat::H265;
};
struct AV1Provider
{
  static constexpr const char * value = "libaom-av1";
  static constexpr const common::ImageFormat format = common::ImageFormat::AV1;
};

}  // namespace accelerated_image_processor::decompression
