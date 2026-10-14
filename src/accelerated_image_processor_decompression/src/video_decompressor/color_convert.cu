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

#include "color_convert.hpp"

namespace accelerated_image_processor::decompression
{
namespace
{
// ITU-R BT.709 full-range (0-255) YCbCr -> R'G'B' coefficients, derived from Kr = 0.2126,
// Kb = 0.0722: Cr->R = 2*(1-Kr), Cb->B = 2*(1-Kb), Cb->G = -2*(1-Kb)*Kb/Kg, Cr->G =
// -2*(1-Kr)*Kr/Kg with Kg = 1-Kr-Kb.
constexpr float kCrToR = 1.5748F;
constexpr float kCbToG = -0.1873F;
constexpr float kCrToG = -0.4681F;
constexpr float kCbToB = 1.8556F;

// Clamp to the 8-bit range and round to nearest (round-to-nearest is the correct quantization of
// the BT.709 matrix output; truncation would add a systematic ~0.5 bias on chroma-bearing pixels).
__device__ inline unsigned char clamp_u8(float value)
{
  return static_cast<unsigned char>(fminf(fmaxf(value, 0.0F), 255.0F) + 0.5F);
}

__device__ inline void write_bgr(
  unsigned char * bgr, int bgr_pitch, int x, int y, float luma, float cb, float cr)
{
  const float r = luma + kCrToR * cr;
  const float g = luma + kCbToG * cb + kCrToG * cr;
  const float b = luma + kCbToB * cb;
  unsigned char * pixel = bgr + static_cast<long>(y) * bgr_pitch + static_cast<long>(x) * 3;
  pixel[0] = clamp_u8(b);
  pixel[1] = clamp_u8(g);
  pixel[2] = clamp_u8(r);
}

__global__ void nv12_bt709_kernel(
  const unsigned char * y_plane, const unsigned char * uv_plane, int pitch, unsigned char * bgr,
  int bgr_pitch, int width, int height)
{
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= width || y >= height) {
    return;
  }
  const float luma = y_plane[static_cast<long>(y) * pitch + x];
  // 4:2:0: one interleaved Cb/Cr pair per 2x2 luma block.
  const long uv_offset = static_cast<long>(y / 2) * pitch + (x / 2) * 2;
  const float cb = static_cast<float>(uv_plane[uv_offset]) - 128.0F;
  const float cr = static_cast<float>(uv_plane[uv_offset + 1]) - 128.0F;
  write_bgr(bgr, bgr_pitch, x, y, luma, cb, cr);
}

__global__ void yuv444_bt709_kernel(
  const unsigned char * y_plane, const unsigned char * cb_plane, const unsigned char * cr_plane,
  int pitch, unsigned char * bgr, int bgr_pitch, int width, int height)
{
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= width || y >= height) {
    return;
  }
  const long offset = static_cast<long>(y) * pitch + x;
  const float luma = y_plane[offset];
  const float cb = static_cast<float>(cb_plane[offset]) - 128.0F;
  const float cr = static_cast<float>(cr_plane[offset]) - 128.0F;
  write_bgr(bgr, bgr_pitch, x, y, luma, cb, cr);
}

dim3 grid_for(int width, int height, const dim3 & block)
{
  return dim3((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
}
}  // namespace

cudaError_t nv12_to_bgr_bt709_full(
  const uint8_t * y, const uint8_t * uv, int pitch, uint8_t * bgr, int bgr_pitch, int width,
  int height, cudaStream_t stream)
{
  const dim3 block(16, 16);
  nv12_bt709_kernel<<<grid_for(width, height, block), block, 0, stream>>>(
    y, uv, pitch, bgr, bgr_pitch, width, height);
  return cudaGetLastError();
}

cudaError_t yuv444_to_bgr_bt709_full(
  const uint8_t * y, const uint8_t * cb, const uint8_t * cr, int pitch, uint8_t * bgr,
  int bgr_pitch, int width, int height, cudaStream_t stream)
{
  const dim3 block(16, 16);
  yuv444_bt709_kernel<<<grid_for(width, height, block), block, 0, stream>>>(
    y, cb, cr, pitch, bgr, bgr_pitch, width, height);
  return cudaGetLastError();
}
}  // namespace accelerated_image_processor::decompression
