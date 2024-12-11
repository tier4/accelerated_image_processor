#include <cstdio>
#include <cstring>

#include <rclcpp/rclcpp.hpp>

#include "accelerator/jpeg_compressor.hpp"

#if defined(JETSON_AVAILABLE) || defined(NVJPEG_AVAILABLE)
#include <nppi_color_conversion.h>
#include <nppi_data_exchange_and_initialization.h>
#endif

#define TEST_ERROR(cond, str) if(cond) { \
                                        fprintf(stderr, "%s\n", str); }

#define CHECK_CUDA(status)       \
    if (status != cudaSuccess) {                                        \
        std::cerr << "CUDA error: " << cudaGetErrorName(status)         \
                  << " (" << __FILE__ << ", " << __LINE__ << ")" << std::endl; \
    }

#define CHECK_NVJPEG(call)                                              \
    {                                                                   \
        nvjpegStatus_t _e = (call);                                     \
        if (_e != NVJPEG_STATUS_SUCCESS) {                              \
            std::cerr << "NVJPEG failure: \'#" << _e << "\' at "        \
                      << __FILE__<< ":" << __LINE__ << std::endl;       \
            exit(1);                                                    \
        }                                                               \
    }

namespace JpegCompressor {

#ifdef TURBOJPEG_AVAILABLE
CPUCompressor::CPUCompressor()
    : jpegBuf_(nullptr), size_(0) {
    handle_ = tjInitCompress();
}

CPUCompressor::~CPUCompressor() {
    if (jpegBuf_)
        tjFree(jpegBuf_);
    tjDestroy(handle_);
}

CompressedImage::UniquePtr CPUCompressor::compress(const Image &msg, int quality, int format, int sampling) {
    CompressedImage::UniquePtr compressed_msg = std::make_unique<CompressedImage>();
    compressed_msg->header = msg.header;
    compressed_msg->format = "jpeg";

    if (jpegBuf_) {
        tjFree(jpegBuf_);
        jpegBuf_ = nullptr;
    }

    int tjres = tjCompress2(handle_,
                            msg.data.data(),
                            msg.width,
                            0,
                            msg.height,
                            format,
                            &jpegBuf_,
                            &size_,
                            sampling,
                            quality,
                            TJFLAG_FASTDCT);

    TEST_ERROR(tjres != 0, tjGetErrorStr2(handle_));

    compressed_msg->data.resize(size_);
    memcpy(compressed_msg->data.data(), jpegBuf_, size_);

    return compressed_msg;
}
#endif

#ifdef JETSON_AVAILABLE
JetsonCompressor::JetsonCompressor(std::string name)
        : stream_(cuda::device::current::get().create_stream(cuda::stream::sync)) {
    encoder_ = NvJPEGEncoder::createJPEGEncoder(name.c_str());
}

JetsonCompressor::~JetsonCompressor() {
    delete encoder_;

    nppiFree(dev_image_);
    for (auto& p : dev_yuv_) {
      nppiFree(p);
    }
}

CompressedImage::UniquePtr JetsonCompressor::compress(const Image &msg, int quality, ImageFormat format) {
    CompressedImage::UniquePtr compressed_msg = std::make_unique<CompressedImage>();
    compressed_msg->header = msg.header;
    compressed_msg->format = "jpeg";

    int width = msg.width;
    int height = msg.height;
    const auto &img = msg.data;

    if (image_size < img.size()) {
      // Allocate Npp8u buffers
      dev_image_ = nppiMalloc_8u_C3(width, height, &dev_image_step_bytes_);
      image_size = img.size();

      dev_yuv_[0] = nppiMalloc_8u_C1(width, height, &dev_yuv_step_bytes_[0]); // Y
      dev_yuv_[1] = nppiMalloc_8u_C1(width/2, height/2, &dev_yuv_step_bytes_[1]); // U
      dev_yuv_[2] = nppiMalloc_8u_C1(width/2, height/2, &dev_yuv_step_bytes_[2]); // V

      // Fill elements of  nppStreamContext
      {
        npp_stream_context_.hStream = stream_.handle();
        cudaGetDevice(&npp_stream_context_.nCudaDeviceId);
        cudaDeviceProp dev_prop;
        cudaGetDeviceProperties(&dev_prop, npp_stream_context_.nCudaDeviceId);
        npp_stream_context_.nMultiProcessorCount = dev_prop.multiProcessorCount;
        npp_stream_context_.nMaxThreadsPerMultiProcessor = dev_prop.maxThreadsPerMultiProcessor;
        npp_stream_context_.nMaxThreadsPerBlock = dev_prop.maxThreadsPerBlock;
        npp_stream_context_.nSharedMemPerBlock = dev_prop.sharedMemPerBlock;
        cudaDeviceGetAttribute(&npp_stream_context_.nCudaDevAttrComputeCapabilityMajor,
                               cudaDevAttrComputeCapabilityMajor, npp_stream_context_.nCudaDeviceId);
        cudaDeviceGetAttribute(&npp_stream_context_.nCudaDevAttrComputeCapabilityMinor,
                               cudaDevAttrComputeCapabilityMinor, npp_stream_context_.nCudaDeviceId);
        cudaStreamGetFlags(npp_stream_context_.hStream, &npp_stream_context_.nStreamFlags);
      }

      buffer_.emplace(V4L2_PIX_FMT_YUV420M, width, height, 0);
      TEST_ERROR(buffer_->allocateMemory() != 0, "NvBuffer allocation failed");

      encoder_->setCropRect(0, 0, width, height);
    }

    TEST_ERROR(cudaMemcpy2DAsync(static_cast<void*>(dev_image_), dev_image_step_bytes_,
                                 static_cast<const void*>(img.data()), msg.step,
                                 msg.step * sizeof(Npp8u),
                                 msg.height, cudaMemcpyHostToDevice, stream_.handle()) != cudaSuccess,
               "2D memory allocation failed");
    NppiSize roi = {static_cast<int>(msg.width), static_cast<int>(msg.height)};
    if (format == ImageFormat::BGR) {
        const int order[3] = {2, 1, 0};
        TEST_ERROR(nppiSwapChannels_8u_C3IR_Ctx(dev_image_, dev_image_step_bytes_, roi, order, npp_stream_context_) != NPP_SUCCESS,
                "failed to convert bgr8 to rgb8"
        );
    }
    TEST_ERROR(nppiRGBToYUV420_8u_C3P3R_Ctx(dev_image_, dev_image_step_bytes_,
                                            dev_yuv_.data(), dev_yuv_step_bytes_.data(), roi,
                                            npp_stream_context_) != NPP_SUCCESS,
                "failed to convert rgb8 to yuv420");

    NvBuffer::NvBufferPlane &plane_y = buffer_->planes[0];
    NvBuffer::NvBufferPlane &plane_u = buffer_->planes[1];
    NvBuffer::NvBufferPlane &plane_v = buffer_->planes[2];
    TEST_ERROR(cudaMemcpy2DAsync(plane_y.data, plane_y.fmt.stride,
                                 dev_yuv_[0], dev_yuv_step_bytes_[0], width, height,
                                 cudaMemcpyDeviceToHost, stream_.handle()) != cudaSuccess,
               "memory copy from Device to Host for Y plane failed");
    TEST_ERROR(cudaMemcpy2DAsync(plane_u.data, plane_u.fmt.stride,
                                 dev_yuv_[1], dev_yuv_step_bytes_[1], width/2, height/2,
                                 cudaMemcpyDeviceToHost, stream_.handle()) != cudaSuccess,
               "memory copy from Device to Host for U plane failed");
    TEST_ERROR(cudaMemcpy2DAsync(plane_v.data, plane_v.fmt.stride,
                                 dev_yuv_[2], dev_yuv_step_bytes_[2], width/2, height/2,
                                 cudaMemcpyDeviceToHost, stream_.handle()) != cudaSuccess,
               "memory copy from Device to Host for V plane failed");
    stream_.synchronize();

    size_t out_buf_size = width * height * 3 / 2;
    unsigned char * out_data = new unsigned char[out_buf_size];

    TEST_ERROR(
        encoder_->encodeFromBuffer(buffer_.value(), JCS_YCbCr, &out_data,
                                   out_buf_size, quality),
        "NvJpeg Encoder Error");

    compressed_msg->data.resize(static_cast<size_t>(out_buf_size / sizeof(uint8_t)));
    memcpy(compressed_msg->data.data(), out_data, out_buf_size);

    delete[] out_data;
    out_data = nullptr;
    return compressed_msg;
}

void JetsonCompressor::setCudaStream(cuda::stream::handle_t &raw_cuda_stream) {
  stream_ = cuda::stream::wrap(cuda::device::current::get().id(),
                               cuda::context::current::get().handle(),
                               raw_cuda_stream);
}
#endif

#ifdef NVJPEG_AVAILABLE
NVJPEGCompressor::NVJPEGCompressor() {
    CHECK_CUDA(cudaStreamCreate(&stream_));
    // CHECK_NVJPEG(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, NULL, NULL, NVJPEG_FLAGS_DEFAULT, &handle_))
    CHECK_NVJPEG(nvjpegCreateSimple(&handle_));
    CHECK_NVJPEG(nvjpegEncoderStateCreate(handle_, &state_, stream_));
    CHECK_NVJPEG(nvjpegEncoderParamsCreate(handle_, &params_, stream_));

    nvjpegEncoderParamsSetSamplingFactors(params_, NVJPEG_CSS_420, stream_);

    std::memset(&nv_image_, 0, sizeof(nv_image_));
}

NVJPEGCompressor::~NVJPEGCompressor() {
    CHECK_NVJPEG(nvjpegEncoderParamsDestroy(params_));
    CHECK_NVJPEG(nvjpegEncoderStateDestroy(state_));
    CHECK_NVJPEG(nvjpegDestroy(handle_));
    CHECK_CUDA(cudaStreamDestroy(stream_));
}

CompressedImage::UniquePtr NVJPEGCompressor::compress(const Image &msg, int quality, ImageFormat format) {
    CompressedImage::UniquePtr compressed_msg = std::make_unique<CompressedImage>();
    compressed_msg->header = msg.header;
    compressed_msg->format = "jpeg";

    nvjpegEncoderParamsSetQuality(params_, quality, stream_);

    nvjpegInputFormat_t input_format;
    if (format == ImageFormat::RGB) {
        input_format = NVJPEG_INPUT_RGBI;
    } else if (format == ImageFormat::BGR) {
        input_format = NVJPEG_INPUT_BGRI;
    } else {
        std::cerr << "Specified ImageFormat is not supported" << std::endl;
    }
    setNVImage(msg);
    CHECK_NVJPEG(nvjpegEncodeImage(handle_, state_, params_, &nv_image_, input_format,
                                   msg.width, msg.height, stream_));

    unsigned long out_buf_size = 0;

    CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(handle_, state_, NULL, &out_buf_size, stream_));
    compressed_msg->data.resize(out_buf_size);
    CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(handle_, state_, compressed_msg->data.data(),
                                               &out_buf_size, stream_));

    CHECK_CUDA(cudaStreamSynchronize(stream_));

    return compressed_msg;
}

void NVJPEGCompressor::setNVImage(const Image &msg) {
    if (nv_image_.channel[0] == nullptr) {
        CHECK_CUDA(cudaMallocAsync((void **)&nv_image_.channel[0], msg.data.size(), stream_));
    }

    CHECK_CUDA(cudaMemsetAsync(nv_image_.channel[0], 0, msg.data.size(), stream_));

    CHECK_CUDA(cudaMemcpyAsync(nv_image_.channel[0], msg.data.data(), msg.data.size(),
                               cudaMemcpyHostToDevice, stream_));

    // int channels = image.size() / (image.width * image.height);
    int channels = 3;

    // Assuming RGBI/BGRI
    nv_image_.pitch[0] = msg.width * channels;
}

void NVJPEGCompressor::setCudaStream(const cudaStream_t &raw_cuda_stream) {
    CHECK_CUDA(cudaStreamDestroy(stream_));
    stream_ = raw_cuda_stream;
}
#endif

} // namespace JpegCompressor
