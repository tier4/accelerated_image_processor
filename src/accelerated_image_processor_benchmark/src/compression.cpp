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

#include "conversion.hpp"
#include "observer.hpp"

#include <accelerated_image_processor_common/datatype.hpp>
#include <accelerated_image_processor_compression/builder.hpp>

#include <iostream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace accelerated_image_processor::benchmark
{
namespace
{
struct Arguments
{
  compression::CompressionType type = compression::CompressionType::JPEG;
  int height = 1920;
  int width = 1080;

  int quality = 90;

  int warmup = 10;
  int iterations = 100;

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
     << "  --type JPEG|VIDEO    Compression type. (default: JPEG)\n"
     << "  --width N            Input width. (default: 1920)\n"
     << "  --height N           Input height. (default: 1080)\n"
     << "  --quality N          JPEG quality [1..100]. (default: 90)\n"
     << "  --warmup N           Warmup iterations (each iteration encodes batch images). (default: "
        "10)\n"
     << "  --iterations N       Measured iterations (each iteration encodes batch images). "
        "(default: 100)\n"
     << "  --seed N             RNG seed for deterministic synthetic input. (default: 1)\n"
     << "  --quiet              Suppress human-readable stderr logs.\n"
     << "  --help               Show this help.\n";
}

static Arguments parse_args(int argc, char ** argv)
{
  Arguments arguments;
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
    } else if (key == "--type") {
      arguments.type = compression::to_compression_type(require_value("--type"));
    } else if (key == "--width") {
      arguments.width = to_int(require_value("--width"), "width");
    } else if (key == "--height") {
      arguments.height = to_int(require_value("--height"), "height");
    } else if (key == "--quality") {
      arguments.quality = to_int(require_value("--quality"), "quality");
    } else if (key == "--warmup") {
      arguments.warmup = to_int(require_value("--warmup"), "warmup");
    } else if (key == "--iterations") {
      arguments.iterations = to_int(require_value("--iterations"), "iterations");
    } else if (key == "--seed") {
      arguments.seed = to_u64(require_value("--seed"), "seed");
    } else if (key == "--quiet") {
      arguments.quiet = true;
    } else {
      std::ostringstream oss;
      oss << "Unknown option: " << key;
      throw std::runtime_error(oss.str());
    }
  }

  if (arguments.width <= 0 || arguments.height <= 0) {
    throw std::runtime_error("width/height must be positive");
  }
  if (arguments.quality < 1 || arguments.quality > 100) {
    throw std::runtime_error("quality must be in [1..100]");
  }
  if (arguments.warmup < 0 || arguments.iterations <= 0) {
    throw std::runtime_error("warmup and iterations must be >= 0");
  }

  return arguments;
}
}  // namespace

int main(int argc, char ** argv)
{
  try {
    const Arguments arguments = parse_args(argc, argv);

    auto compressor = compression::create_compressor(arguments.type);
    if (!compressor) {
      throw std::runtime_error("Failed to create compressor");
    }

    // Observe output sizes via postprocess callback
    ProcessObserver observer;
    compressor->register_postprocess<ProcessObserver, &ProcessObserver::on_image>(&observer);

    auto try_processing = [&](int iter_idx) {};

    // Warmup
    for (int i = 0; i < arguments.warmup; ++i) {
      try_processing(i);
    }

    // Reset observer and run iterations
    observer.clear();
    for (int i = 0; i < arguments.iterations; ++i) {
      observer.tic();
      try_processing(i);
    }

    const auto avg_bytes = observer.average_bytes();
    std::cout << observer.percentile_ms() << std::endl;
  } catch (const std::exception & e) {
    std::cerr << "Error: " << e.what() << std::endl;
    std::exit(1);
  }
}
}  // namespace accelerated_image_processor::benchmark
