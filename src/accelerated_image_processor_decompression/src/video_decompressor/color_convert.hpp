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

#include <cuda_runtime.h>

#include <cstdint>

namespace accelerated_image_processor::decompression
{
/**
 * @brief Convert an NVDEC NV12 surface to packed BGR with the standard ITU-R BT.709 full-range
 * matrix.
 *
 * The companion compression package encodes full-range BT.709 YCbCr. NPP's fixed NV12->BGR
 * converters do not correctly invert that (709HDTV desaturates chroma, 709CSC crushes the luma
 * range), so this applies the exact ITU-R BT.709 full-range equations instead.
 *
 * @param y        Device pointer to the luma plane at the crop origin.
 * @param uv        Device pointer to the interleaved Cb/Cr plane at the crop origin.
 * @param pitch     Shared row pitch (bytes) of the NV12 planes.
 * @param bgr        Device pointer to the destination packed BGR buffer.
 * @param bgr_pitch  Row pitch (bytes) of the destination buffer.
 * @param width      Output width in pixels (must be even).
 * @param height     Output height in pixels (must be even).
 * @param stream     CUDA stream to launch on.
 * @return cudaSuccess on success, otherwise the first CUDA error observed.
 */
cudaError_t nv12_to_bgr_bt709_full(
  const uint8_t * y, const uint8_t * uv, int pitch, uint8_t * bgr, int bgr_pitch, int width,
  int height, cudaStream_t stream);

/**
 * @brief Convert a planar 4:4:4 YCbCr surface to packed BGR with the ITU-R BT.709 full-range
 * matrix.
 *
 * Counterpart of nv12_to_bgr_bt709_full() for the full-resolution 4:4:4 layout (three separate
 * planes sharing the same row pitch).
 */
cudaError_t yuv444_to_bgr_bt709_full(
  const uint8_t * y, const uint8_t * cb, const uint8_t * cr, int pitch, uint8_t * bgr,
  int bgr_pitch, int width, int height, cudaStream_t stream);
}  // namespace accelerated_image_processor::decompression
