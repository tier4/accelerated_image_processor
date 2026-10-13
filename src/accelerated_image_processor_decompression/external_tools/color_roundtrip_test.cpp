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

#include <accelerated_image_processor_common/datatype.hpp>
#include <accelerated_image_processor_compression/video_compressor.hpp>
#include <accelerated_image_processor_decompression/video_decompressor.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace
{
namespace common = accelerated_image_processor::common;
namespace compression = accelerated_image_processor::compression;
namespace decompression = accelerated_image_processor::decompression;

constexpr uint32_t kWidth = 640;
constexpr uint32_t kHeight = 480;
constexpr uint32_t kSwatchWidth = 80;
constexpr uint32_t kSwatchHeight = 160;
constexpr std::size_t kFrameCount = 8;
constexpr double kMaeLimit = 10.0;
constexpr uint8_t kAvPacketFlagKey = 0x1;

struct Swatch
{
  const char * name;
  std::array<uint8_t, 3> bgr;
  uint32_t x;
  uint32_t y;
};

constexpr std::array<Swatch, 13> kSwatches{{
  {"White", {255, 255, 255}, 0, 0},
  {"Black", {0, 0, 0}, 80, 0},
  {"Gray", {128, 128, 128}, 160, 0},
  // Shadow / highlight grays to catch luma range crushing (black crush / white clip).
  {"Gray8", {8, 8, 8}, 240, 0},
  {"Gray16", {16, 16, 16}, 320, 0},
  {"Gray240", {240, 240, 240}, 400, 0},
  {"Gray248", {248, 248, 248}, 480, 0},
  {"Red", {0, 0, 255}, 0, 160},
  {"Green", {0, 255, 0}, 80, 160},
  {"Blue", {255, 0, 0}, 160, 160},
  {"Cyan", {255, 255, 0}, 0, 320},
  {"Magenta", {255, 0, 255}, 80, 320},
  {"Yellow", {0, 255, 255}, 160, 320},
}};

struct DecodedCollector
{
  void collect(const common::Image & image) { frames.push_back(image); }
  std::vector<common::Image> frames;
};

std::string exception_message(const std::exception_ptr & exception)
{
  if (!exception) {
    return "no exception was captured";
  }
  try {
    std::rethrow_exception(exception);
  } catch (const std::exception & error) {
    return error.what();
  } catch (...) {
    return "unknown non-standard exception";
  }
}

common::Image make_test_image()
{
  common::Image image{};
  image.frame_id = "color_roundtrip";
  image.timestamp = 0;
  image.width = kWidth;
  image.height = kHeight;
  image.step = kWidth * 3U;
  image.format = common::ImageFormat::RAW;
  image.encoding = common::ImageEncoding::BGR;
  image.is_bigendian = false;
  image.data.assign(static_cast<std::size_t>(image.step) * image.height, 0);

  for (const auto & swatch : kSwatches) {
    for (uint32_t y = swatch.y; y < swatch.y + kSwatchHeight; ++y) {
      for (uint32_t x = swatch.x; x < swatch.x + kSwatchWidth; ++x) {
        const std::size_t offset = static_cast<std::size_t>(y) * image.step + x * 3U;
        image.data[offset] = swatch.bgr[0];
        image.data[offset + 1] = swatch.bgr[1];
        image.data[offset + 2] = swatch.bgr[2];
      }
    }
  }
  return image;
}

std::array<double, 3> roi_mean(const common::Image & image, const Swatch & swatch)
{
  std::array<uint64_t, 3> sums{};
  for (uint32_t y = swatch.y; y < swatch.y + kSwatchHeight; ++y) {
    for (uint32_t x = swatch.x; x < swatch.x + kSwatchWidth; ++x) {
      const std::size_t offset = static_cast<std::size_t>(y) * image.step + x * 3U;
      for (std::size_t channel = 0; channel < sums.size(); ++channel) {
        sums[channel] += image.data[offset + channel];
      }
    }
  }

  constexpr double pixel_count = static_cast<double>(kSwatchWidth) * kSwatchHeight;
  return {
    sums[0] / pixel_count,
    sums[1] / pixel_count,
    sums[2] / pixel_count,
  };
}

void validate_decoded_image(const common::Image & image, std::size_t frame_index)
{
  if (
    image.width != kWidth || image.height != kHeight || image.step != kWidth * 3U ||
    image.format != common::ImageFormat::RAW || image.encoding != common::ImageEncoding::BGR ||
    image.data.size() != static_cast<std::size_t>(kWidth) * kHeight * 3U) {
    throw std::runtime_error(
      "Decoded frame " + std::to_string(frame_index) +
      " has unexpected BGR image metadata or size");
  }
}

int run()
{
  std::cout << "NVENC AV1 -> NVDEC BGR color round-trip\n";
  std::cout << "Resolution: " << kWidth << 'x' << kHeight << ", input frames: " << kFrameCount
            << ", threshold: each channel MAE < " << kMaeLimit << "\n";

  auto compressor = compression::make_nvenc_av1_compressor();
  if (!compressor) {
    throw std::runtime_error("make_nvenc_av1_compressor() returned nullptr");
  }
  const auto [encoder_available, encoder_message] =
    compressor->validate_compression_type_compatibility();
  if (!encoder_available) {
    throw std::runtime_error("NVENC AV1 compatibility check failed: " + encoder_message);
  }
  std::cout << "NVENC AV1 compatibility check: PASS\n";

  common::Image input = make_test_image();
  std::vector<common::Image> packets;
  packets.reserve(kFrameCount);
  for (std::size_t frame = 0; frame < kFrameCount; ++frame) {
    input.timestamp = static_cast<int64_t>(frame) * 100000000;
    const std::optional<common::Image> encoded = compressor->process(input);
    if (!encoded) {
      throw std::runtime_error(
        "Encoding frame " + std::to_string(frame) +
        " failed: " + exception_message(compressor->exception_ptr()));
    }
    packets.push_back(*encoded);
    const bool keyframe =
      packets.back().flags.has_value() && ((*packets.back().flags & kAvPacketFlagKey) != 0);
    std::cout << "Encoded packet " << frame << ": bytes=" << packets.back().data.size() << ", pts=";
    if (packets.back().pts) {
      std::cout << *packets.back().pts;
    } else {
      std::cout << "unset";
    }
    std::cout << ", flags=";
    if (packets.back().flags) {
      std::cout << static_cast<unsigned int>(*packets.back().flags);
    } else {
      std::cout << "unset";
    }
    std::cout << ", keyframe=" << (keyframe ? "yes" : "no") << '\n';
  }

  if (!packets.front().flags.has_value() || ((*packets.front().flags & kAvPacketFlagKey) == 0)) {
    throw std::runtime_error("The first encoded packet was not flagged as a keyframe");
  }

  // Debug dump: concatenate the elementary AV1 OBU stream for an independent ffmpeg cross-check.
  if (const char * obu_path = std::getenv("ROUNDTRIP_DUMP_OBU")) {
    std::ofstream obu(obu_path, std::ios::binary);
    for (const auto & packet : packets) {
      obu.write(
        reinterpret_cast<const char *>(packet.data.data()),
        static_cast<std::streamsize>(packet.data.size()));
    }
    std::cout << "Dumped " << packets.size() << " packets to " << obu_path << '\n';
  }

  auto decompressor = decompression::make_nvdec_video_decompressor();
  if (!decompressor) {
    throw std::runtime_error("make_nvdec_video_decompressor() returned nullptr");
  }
  DecodedCollector collector;
  decompressor->register_postprocess<DecodedCollector, &DecodedCollector::collect>(&collector);

  for (const auto & packet : packets) {
    decompressor->process(packet);
    if (decompressor->exception_ptr()) {
      throw std::runtime_error(
        "NVDEC failed while processing encoded packet: " +
        exception_message(decompressor->exception_ptr()));
    }
  }
  common::Image end_of_stream{};
  end_of_stream.format = common::ImageFormat::AV1;
  decompressor->process(end_of_stream);
  if (decompressor->exception_ptr()) {
    throw std::runtime_error(
      "NVDEC failed while flushing: " + exception_message(decompressor->exception_ptr()));
  }

  std::cout << "Decoded frames collected after flush: " << collector.frames.size() << '\n';
  if (collector.frames.empty()) {
    throw std::runtime_error("NVDEC produced no decoded frames");
  }

  // Debug dump: the project-decoded BGR of the first frame for an independent ffmpeg cross-check.
  if (const char * bgr_path = std::getenv("ROUNDTRIP_DUMP_BGR")) {
    std::ofstream bgr(bgr_path, std::ios::binary);
    bgr.write(
      reinterpret_cast<const char *>(collector.frames.front().data.data()),
      static_cast<std::streamsize>(collector.frames.front().data.size()));
    std::cout << "Dumped decoded frame 0 BGR to " << bgr_path << '\n';
  }
  if (collector.frames.size() != packets.size()) {
    throw std::runtime_error(
      "Decoded frame count " + std::to_string(collector.frames.size()) +
      " does not match encoded packet count " + std::to_string(packets.size()));
  }

  bool overall_pass = true;
  std::cout << std::fixed << std::setprecision(2);
  for (std::size_t frame = 0; frame < collector.frames.size(); ++frame) {
    const auto & decoded = collector.frames[frame];
    validate_decoded_image(decoded, frame);
    std::cout << "\nFrame " << frame << " (pts=";
    if (decoded.pts) {
      std::cout << *decoded.pts;
    } else {
      std::cout << "unset";
    }
    std::cout << ")\n";
    std::cout
      << "Swatch  | Known BGR       | Decoded BGR (mean)       | MAE B | MAE G | MAE R | Result\n";
    std::cout
      << "--------+-----------------+--------------------------+-------+-------+-------+-------\n";
    for (const auto & swatch : kSwatches) {
      const auto mean = roi_mean(decoded, swatch);
      const std::array<double, 3> mae{
        std::abs(mean[0] - swatch.bgr[0]),
        std::abs(mean[1] - swatch.bgr[1]),
        std::abs(mean[2] - swatch.bgr[2]),
      };
      const bool pass = mae[0] < kMaeLimit && mae[1] < kMaeLimit && mae[2] < kMaeLimit;
      overall_pass = overall_pass && pass;

      std::ostringstream known;
      known << '(' << static_cast<unsigned int>(swatch.bgr[0]) << ", "
            << static_cast<unsigned int>(swatch.bgr[1]) << ", "
            << static_cast<unsigned int>(swatch.bgr[2]) << ')';
      std::ostringstream measured;
      measured << std::fixed << std::setprecision(2) << '(' << mean[0] << ", " << mean[1] << ", "
               << mean[2] << ')';

      std::cout << std::left << std::setw(8) << swatch.name << " | " << std::setw(15) << known.str()
                << " | " << std::setw(24) << measured.str() << " | " << std::right << std::setw(5)
                << mae[0] << " | " << std::setw(5) << mae[1] << " | " << std::setw(5) << mae[2]
                << " | " << (pass ? "PASS" : "FAIL") << '\n';
    }
  }

  std::cout << "\nOverall: " << (overall_pass ? "PASS" : "FAIL") << '\n';
  return overall_pass ? 0 : 1;
}
}  // namespace

int main()
{
  try {
    return run();
  } catch (const std::exception & error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    std::cerr << "Overall: FAIL\n";
    return 2;
  }
}
