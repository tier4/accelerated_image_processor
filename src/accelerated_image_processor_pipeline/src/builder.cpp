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

#include "accelerated_image_processor_pipeline/builder.hpp"

#include "accelerated_image_processor_pipeline/rectifier.hpp"

#include <memory>
#include <utility>

namespace accelerated_image_processor::pipeline
{
std::unique_ptr<Rectifier> create_rectifier()
{
#ifdef NPP_AVAILABLE
  return std::make_unique<NPPRectifier>();
#elif OPENCV_CUDA_AVAILABLE
  return std::make_unique<OpenCVCUDARectifier>();
#else
  return std::make_unique<CpuRectifier>();
#endif
}
}  // namespace accelerated_image_processor::pipeline
