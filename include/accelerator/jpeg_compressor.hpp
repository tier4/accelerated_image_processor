#pragma once

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <string>

#ifdef TURBOJPEG_AVAILABLE
#include <turbojpeg.h>
#endif

#ifdef JETSON_AVAILABLE
#include <NvJpegEncoder.h>
#include <cuda/api.hpp>
#include "color_space.hpp"
#include <nppi_support_functions.h>
#endif

#ifdef NVJPEG_AVAILABLE
#include <nvjpeg.h>
#endif

class NvJPEGEncoder;

namespace JpegCompressor {
using Image = sensor_msgs::msg::Image;
using CompressedImage = sensor_msgs::msg::CompressedImage;

enum class ImageFormat {
    RGB,
    BGR
};

#ifdef TURBOJPEG_AVAILABLE
class CPUCompressor {
public:
    CPUCompressor();
    ~CPUCompressor();

    CompressedImage::UniquePtr compress(const Image &msg, int quality = 90, int format = TJPF_RGB, int sampling = TJ_420);
private:
    tjhandle handle_;
    unsigned char *jpegBuf_;
    unsigned long size_;
};
#endif

#ifdef JETSON_AVAILABLE
class JetsonCompressor {
public:
    JetsonCompressor(std::string name);
    ~JetsonCompressor();

    CompressedImage::UniquePtr compress(const Image &msg, int quality = 90, ImageFormat format = ImageFormat::RGB);
    void setCudaStream(cuda::stream::handle_t &raw_cuda_stream);
private:
    NvJPEGEncoder *encoder_;
    size_t image_size{};
    size_t yuv_size{};
    Npp8u *dev_image_;
    std::array<Npp8u*, 3> dev_yuv_;
    std::array<void*, 3> host_yuv_;
    int dev_image_step_bytes_;
    std::array<int, 3> dev_yuv_step_bytes_;

    cuda::stream_t stream_;
    NppStreamContext npp_stream_context_;
    std::optional<NvBuffer> buffer_;
};
#endif

#ifdef NVJPEG_AVAILABLE
class NVJPEGCompressor {
public:
    NVJPEGCompressor();
    ~NVJPEGCompressor();

    CompressedImage::UniquePtr compress(const Image &msg, int quality = 90, ImageFormat format = ImageFormat::RGB);
    void setCudaStream(const cudaStream_t &raw_cuda_stream);

private:
    // void setNVJPEGParams(int quality, ImageFormat format);
    void setNVImage(const Image &msg);

    cudaStream_t stream_;
    nvjpegHandle_t handle_;
    nvjpegEncoderState_t state_;
    nvjpegEncoderParams_t params_;
    nvjpegInputFormat_t input_format_;
    nvjpegChromaSubsampling_t subsampling_;
    nvjpegImage_t nv_image_;
};
#endif

} // namespace JpegCompressor
