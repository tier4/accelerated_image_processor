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
//
// CLI benchmark for accelerated_image_processor_compression (encode-only).
// - Measures JPEG encode latency/throughput
// - Outputs JSON Lines (JSONL): one record per run
//
// Notes:
// - The compression library selects backend at build time (JETSON_AVAILABLE / NVJPEG_AVAILABLE /
// TURBOJPEG_AVAILABLE).
// - common::BaseProcessor::process() is void; we use a postprocess callback to observe output size.

#include <accelerated_image_processor_common/datatype.hpp>
#include <accelerated_image_processor_compression/builder.hpp>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace aip = accelerated_image_processor;
namespace cc = aip::common;
namespace comp = aip::compression;

namespace
{

struct Args
{
  std::string output_path = "-";  // "-" => stdout
  int width = 1920;
  int height = 1080;
  std::string encoding = "RGB";  // RGB|BGR
  int quality = 90;

  int warmup = 10;
  int iterations = 100;
  int batch = 1;

  uint64_t seed = 1;
  bool quiet = false;
};

static void print_usage(std::ostream & os, const char * argv0)
{
  os << "Usage: " << argv0 << " [options]\n"
     << "\n"
     << "Encode-only benchmark for accelerated_image_processor_compression (JPEG).\n"
     << "\n"
     << "Options:\n"
     << "  --output PATH        Output JSONL path ('-' for stdout). (default: -)\n"
     << "  --width N            Input width. (default: 1920)\n"
     << "  --height N           Input height. (default: 1080)\n"
     << "  --encoding RGB|BGR   Input pixel encoding. (default: RGB)\n"
     << "  --quality N          JPEG quality [1..100]. (default: 90)\n"
     << "  --warmup N           Warmup iterations (each iteration encodes batch images). (default: "
        "10)\n"
     << "  --iterations N       Measured iterations (each iteration encodes batch images). "
        "(default: 100)\n"
     << "  --batch N            Images per iteration. (default: 1)\n"
     << "  --seed N             RNG seed for deterministic synthetic input. (default: 1)\n"
     << "  --quiet              Suppress human-readable stderr logs.\n"
     << "  --help               Show this help.\n";
}

static bool iequals(std::string_view a, std::string_view b)
{
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    const auto ca = static_cast<unsigned char>(a[i]);
    const auto cb = static_cast<unsigned char>(b[i]);
    if (std::toupper(ca) != std::toupper(cb)) return false;
  }
  return true;
}

static int to_int(const std::string & s, const char * name)
{
  char * end = nullptr;
  errno = 0;
  long v = std::strtol(s.c_str(), &end, 10);
  if (errno != 0 || end == s.c_str() || *end != '\0') {
    std::ostringstream oss;
    oss << "Invalid integer for " << name << ": '" << s << "'";
    throw std::runtime_error(oss.str());
  }
  if (v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max()) {
    std::ostringstream oss;
    oss << "Out of range integer for " << name << ": '" << s << "'";
    throw std::runtime_error(oss.str());
  }
  return static_cast<int>(v);
}

static uint64_t to_u64(const std::string & s, const char * name)
{
  char * end = nullptr;
  errno = 0;
  unsigned long long v = std::strtoull(s.c_str(), &end, 10);
  if (errno != 0 || end == s.c_str() || *end != '\0') {
    std::ostringstream oss;
    oss << "Invalid uint64 for " << name << ": '" << s << "'";
    throw std::runtime_error(oss.str());
  }
  return static_cast<uint64_t>(v);
}

static Args parse_args(int argc, char ** argv)
{
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];

    auto require_value = [&](const char * opt) -> std::string {
      if (i + 1 >= argc) {
        std::ostringstream oss;
        oss << "Missing value for " << opt;
        throw std::runtime_error(oss.str());
      }
      return std::string(argv[++i]);
    };

    if (key == "--help" || key == "-h") {
      print_usage(std::cout, argv[0]);
      std::exit(0);
    } else if (key == "--output") {
      a.output_path = require_value("--output");
    } else if (key == "--width") {
      a.width = to_int(require_value("--width"), "width");
    } else if (key == "--height") {
      a.height = to_int(require_value("--height"), "height");
    } else if (key == "--encoding") {
      a.encoding = require_value("--encoding");
    } else if (key == "--quality") {
      a.quality = to_int(require_value("--quality"), "quality");
    } else if (key == "--warmup") {
      a.warmup = to_int(require_value("--warmup"), "warmup");
    } else if (key == "--iterations") {
      a.iterations = to_int(require_value("--iterations"), "iterations");
    } else if (key == "--batch") {
      a.batch = to_int(require_value("--batch"), "batch");
    } else if (key == "--seed") {
      a.seed = to_u64(require_value("--seed"), "seed");
    } else if (key == "--quiet") {
      a.quiet = true;
    } else {
      std::ostringstream oss;
      oss << "Unknown option: " << key;
      throw std::runtime_error(oss.str());
    }
  }

  if (a.width <= 0 || a.height <= 0) {
    throw std::runtime_error("width/height must be positive");
  }
  if (!(iequals(a.encoding, "RGB") || iequals(a.encoding, "BGR"))) {
    throw std::runtime_error("encoding must be RGB or BGR");
  }
  if (a.quality < 1 || a.quality > 100) {
    throw std::runtime_error("quality must be in [1..100]");
  }
  if (a.warmup < 0 || a.iterations <= 0 || a.batch <= 0) {
    throw std::runtime_error("warmup must be >= 0, iterations and batch must be > 0");
  }

  return a;
}

static std::string backend_string()
{
#if defined(JETSON_AVAILABLE)
  return "JETSON";
#elif defined(NVJPEG_AVAILABLE)
  return "NVJPEG";
#elif defined(TURBOJPEG_AVAILABLE)
  return "CPU";
#else
  return "UNKNOWN";
#endif
}

// Minimal JSON string escape (suitable for file/program identifiers).
static std::string json_escape(std::string_view s)
{
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          // control char
          std::ostringstream oss;
          oss << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(static_cast<unsigned char>(c));
          out += oss.str();
        } else {
          out += c;
        }
        break;
    }
  }
  return out;
}

static void set_quality_via_parameters(comp::Compressor & compressor, int quality)
{
  // The JPEG compressor class sets a dedicated parameter named "quality".
  // Since Compressor is aliased to common::BaseProcessor, we update via ParameterMap.
  auto & params = compressor.parameters();
  params["quality"] = quality;
}

static cc::ImageEncoding parse_encoding(std::string_view s)
{
  if (iequals(s, "RGB")) return cc::ImageEncoding::RGB;
  if (iequals(s, "BGR")) return cc::ImageEncoding::BGR;
  throw std::runtime_error("Unsupported encoding: " + std::string(s));
}

static cc::Image make_synthetic_image(
  int width, int height, cc::ImageEncoding enc, uint64_t seed, uint64_t index)
{
  // Deterministic, moderately "image-like" pattern with some noise, to avoid compressing trivially.
  // 3 channels interleaved, uint8.
  cc::Image img;
  img.frame_id = "bench";
  img.timestamp = 0;
  img.width = static_cast<uint32_t>(width);
  img.height = static_cast<uint32_t>(height);
  img.step = static_cast<uint32_t>(width * 3);
  img.encoding = enc;
  img.format = cc::ImageFormat::RAW;
  img.data.resize(static_cast<size_t>(height) * static_cast<size_t>(img.step));

  std::mt19937_64 rng(seed ^ (index + 0x9E3779B97F4A7C15ULL));
  std::uniform_int_distribution<int> noise(-12, 12);

  for (int y = 0; y < height; ++y) {
    uint8_t * row = img.data.data() + static_cast<size_t>(y) * img.step;
    for (int x = 0; x < width; ++x) {
      const int base0 = (x * 255) / std::max(1, width - 1);
      const int base1 = (y * 255) / std::max(1, height - 1);
      const int base2 = ((x + y) * 255) / std::max(1, (width + height - 2));
      const int n0 = noise(rng);
      const int n1 = noise(rng);
      const int n2 = noise(rng);

      auto clamp_u8 = [](int v) -> uint8_t {
        if (v < 0) return 0;
        if (v > 255) return 255;
        return static_cast<uint8_t>(v);
      };

      row[x * 3 + 0] = clamp_u8(base0 + n0);
      row[x * 3 + 1] = clamp_u8(base1 + n1);
      row[x * 3 + 2] = clamp_u8(base2 + n2);
    }
  }
  return img;
}

static double percentile_ms(std::vector<double> v, double p)
{
  if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
  std::sort(v.begin(), v.end());
  const double clamped = std::min(1.0, std::max(0.0, p));
  const double pos = clamped * static_cast<double>(v.size() - 1);
  const size_t i0 = static_cast<size_t>(std::floor(pos));
  const size_t i1 = static_cast<size_t>(std::ceil(pos));
  if (i0 == i1) return v[i0];
  const double t = pos - static_cast<double>(i0);
  return v[i0] * (1.0 - t) + v[i1] * t;
}

struct OutputObserver
{
  uint64_t total_bytes = 0;
  uint64_t count = 0;

  void on_image(const cc::Image & out)
  {
    total_bytes += static_cast<uint64_t>(out.data.size());
    count += 1;
  }

  void on_image_nonconst(const cc::Image & out) { on_image(out); }
};

}  // namespace

int main(int argc, char ** argv)
{
  try {
    const Args args = parse_args(argc, argv);

    // Output stream
    std::unique_ptr<std::ostream> owned;
    std::ostream * os = nullptr;
    if (args.output_path == "-" || args.output_path.empty()) {
      os = &std::cout;
    } else {
      auto f = std::make_unique<std::ofstream>(args.output_path, std::ios::out | std::ios::app);
      if (!f->is_open()) {
        throw std::runtime_error("Failed to open output file: " + args.output_path);
      }
      os = f.get();
      owned = std::move(f);
    }

    // Build compressor (backend selected at build-time in compression pkg)
    auto compressor = comp::create_compressor(comp::CompressionType::JPEG);
    if (!compressor) {
      throw std::runtime_error("create_compressor returned null");
    }

    set_quality_via_parameters(*compressor, args.quality);

    // Observe output sizes via postprocess callback
    OutputObserver obs;
    compressor->register_postprocess<OutputObserver, &OutputObserver::on_image_nonconst>(&obs);

    // Pre-generate synthetic inputs to reduce variance from RNG during measurement.
    // Total images to be encoded in measured part: iterations * batch.
    const int total_images = args.iterations * args.batch;
    std::vector<cc::Image> inputs;
    inputs.reserve(static_cast<size_t>(std::max(1, total_images)));

    const cc::ImageEncoding enc = parse_encoding(args.encoding);
    for (int i = 0; i < std::max(1, total_images); ++i) {
      inputs.push_back(make_synthetic_image(args.width, args.height, enc, args.seed, i));
    }

    auto encode_batch = [&](int iter_idx) {
      const int base = iter_idx * args.batch;
      for (int j = 0; j < args.batch; ++j) {
        const int idx = (base + j) % static_cast<int>(inputs.size());
        compressor->process(inputs[static_cast<size_t>(idx)]);
      }
    };

    if (!args.quiet) {
      std::cerr << "[compression_encode_bench] backend=" << backend_string()
                << " quality=" << args.quality << " " << args.width << "x" << args.height
                << " encoding=" << args.encoding << " warmup=" << args.warmup
                << " iterations=" << args.iterations << " batch=" << args.batch << "\n";
    }

    // Warmup
    for (int i = 0; i < args.warmup; ++i) {
      encode_batch(i);
    }

    // Measure per-iteration time (each iteration encodes `batch` images)
    std::vector<double> iter_ms;
    iter_ms.reserve(static_cast<size_t>(args.iterations));

    // Reset observer counters for measured section
    obs.total_bytes = 0;
    obs.count = 0;

    using clock = std::chrono::steady_clock;
    for (int i = 0; i < args.iterations; ++i) {
      const auto t0 = clock::now();
      encode_batch(i);
      const auto t1 = clock::now();

      const auto dt =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0);
      iter_ms.push_back(dt.count());
    }

    const double total_ms =
      std::accumulate(iter_ms.begin(), iter_ms.end(), 0.0, std::plus<double>());
    const double images_total =
      static_cast<double>(args.iterations) * static_cast<double>(args.batch);
    const double avg_iter_ms = total_ms / static_cast<double>(iter_ms.size());
    const double avg_image_ms = total_ms / images_total;
    const double throughput_ips = (images_total / total_ms) * 1000.0;

    const double p50_iter_ms = percentile_ms(iter_ms, 0.50);
    const double p95_iter_ms = percentile_ms(iter_ms, 0.95);

    const double avg_bytes = (obs.count > 0) ? (static_cast<double>(obs.total_bytes) / obs.count)
                                             : std::numeric_limits<double>::quiet_NaN();

    // Emit JSONL (one line)
    // Keep it dependency-free (no JSON library).
    // Schema is stable and easy to parse.
    (*os) << "{"
          << "\"tool\":\"compression_encode_bench\""
          << ",\"codec\":\"JPEG\""
          << ",\"backend\":\"" << json_escape(backend_string()) << "\""
          << ",\"width\":" << args.width << ",\"height\":" << args.height << ",\"encoding\":\""
          << json_escape(args.encoding) << "\""
          << ",\"quality\":" << args.quality << ",\"warmup\":" << args.warmup
          << ",\"iterations\":" << args.iterations << ",\"batch\":" << args.batch
          << ",\"seed\":" << args.seed
          << ",\"measured_images\":" << static_cast<uint64_t>(images_total)
          << ",\"total_time_ms\":" << std::fixed << std::setprecision(6) << total_ms
          << ",\"avg_iter_ms\":" << std::fixed << std::setprecision(6) << avg_iter_ms
          << ",\"p50_iter_ms\":" << std::fixed << std::setprecision(6) << p50_iter_ms
          << ",\"p95_iter_ms\":" << std::fixed << std::setprecision(6) << p95_iter_ms
          << ",\"avg_image_ms\":" << std::fixed << std::setprecision(6) << avg_image_ms
          << ",\"throughput_images_per_sec\":" << std::fixed << std::setprecision(3)
          << throughput_ips << ",\"output_total_bytes\":" << obs.total_bytes
          << ",\"output_avg_bytes\":" << std::fixed << std::setprecision(3) << avg_bytes << "}\n";

    if (!args.quiet) {
      std::cerr << "[compression_encode_bench] done: "
                << "avg_image_ms=" << std::fixed << std::setprecision(3) << avg_image_ms
                << " throughput=" << std::fixed << std::setprecision(1) << throughput_ips
                << " img/s"
                << " avg_bytes=" << std::fixed << std::setprecision(1) << avg_bytes << "\n";
    }

    return 0;
  } catch (const std::exception & e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    std::cerr << "Use --help for usage.\n";
    return 2;
  }
}
