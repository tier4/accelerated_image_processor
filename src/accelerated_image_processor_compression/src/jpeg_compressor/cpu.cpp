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

#include "accelerated_image_processor_compression/jpeg_compressor.hpp"

#include <accelerated_image_processor_common/datatype.hpp>
#include <accelerated_image_processor_common/helper.hpp>

#include <cstring>
#include <memory>
#include <stdexcept>

#ifdef TURBOJPEG_AVAILABLE
#include <turbojpeg.h>
#endif  // TURBOJPEG_AVAILABLE

namespace accelerated_image_processor::compression
{
#ifdef TURBOJPEG_AVAILABLE
/**
 * @brief JPEG compressor using CPU (TurboJPEG) library.
 */
class CpuJPEGCompressor final : public JPEGCompressor
{
public:
  CpuJPEGCompressor() : JPEGCompressor(JPEGBackend::CPU) { handle_ = tjInitCompress(); }
  ~CpuJPEGCompressor() override
  {
    if (buffer_) {
      tjFree(buffer_);
      buffer_ = nullptr;
    }
    tjDestroy(handle_);
  }

private:
  common::Image process_impl(const common::Image & image) override
  {
    int tjpf;
    if (image.encoding == common::ImageEncoding::RGB) {
      tjpf = TJPF_RGB;
    } else if (image.encoding == common::ImageEncoding::BGR) {
      tjpf = TJPF_BGR;
    } else {
      throw std::runtime_error("Unsupported image format");
    }
    constexpr int sampling = TJ_420;

    int result = tjCompress2(
      handle_, image.data.data(), image.width, 0, image.height, tjpf, &buffer_, &size_, sampling,
      this->quality(), TJFLAG_FASTDCT);

    CHECK_ERROR(result != 0, tjGetErrorStr());

    common::Image output;
    output.frame_id = image.frame_id;
    output.timestamp = image.timestamp;
    output.height = image.height;
    output.width = image.width;
    output.step = 0;  // 0 means this value is pointless because it's compressed
    output.encoding = image.encoding;
    output.format = common::ImageFormat::JPEG;
    output.data.resize(size_ / sizeof(uint8_t));
    std::memcpy(output.data.data(), buffer_, size_);

    return output;
  }

  tjhandle handle_ = nullptr;         //!< TurboJPEG handle for JPEG encoding.
  unsigned char * buffer_ = nullptr;  //!< Buffer for JPEG data.
  size_t size_ = 0;                   //!< Size of the buffer for JPEG data.
};

std::unique_ptr<JPEGCompressor> make_cpujpeg_compressor()
{
  return std::make_unique<CpuJPEGCompressor>();
}
#else
std::unique_ptr<JPEGCompressor> make_cpujpeg_compressor()
{
  return nullptr;
}
#endif  // TURBOJPEG_AVAILABLE
}  // namespace accelerated_image_processor::compression
