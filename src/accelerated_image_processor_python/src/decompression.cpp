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

#include "accelerated_image_processor_common/datatype.hpp"
#include "binding.hpp"

#include <accelerated_image_processor_decompression/builder.hpp>
#include <accelerated_image_processor_decompression/video_decompressor.hpp>

#include <boost/python.hpp>

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace bp = boost::python;                 // NOLINT
using namespace accelerated_image_processor;  // NOLINT

namespace
{
/**
 * @brief PythonDecompressorProxy is a proxy class for decompression::Decompressor.
 */
class PythonDecompressorProxy
{
public:
  explicit PythonDecompressorProxy(std::unique_ptr<decompression::Decompressor> decompressor)
  : decompressor_(std::move(decompressor))
  {
    if (decompressor_) {
      decompressor_
        ->register_postprocess<PythonDecompressorProxy, &PythonDecompressorProxy::on_postprocess>(
          this);
    }
  }

  std::optional<common::Image> process(const common::Image & image)
  {
    if (!decompressor_) {
      return std::nullopt;
    }
    return decompressor_->process(image);
  }

  void register_postprocess(const bp::object & callback) { callback_ = callback; }

private:
  void on_postprocess(const common::Image & image)
  {
    if (callback_) {
      callback_(image);
    }
  }

  std::unique_ptr<decompression::Decompressor> decompressor_;
  bp::object callback_;
};
}  // namespace

BOOST_PYTHON_MODULE(accelerated_image_processor_python_decompression)
{
  bp::class_<PythonDecompressorProxy, boost::noncopyable>("Decompressor", bp::no_init)
    .def("process", &python::process_or_none<PythonDecompressorProxy>)
    .def("register_postprocess", &PythonDecompressorProxy::register_postprocess)
    .add_property(
      "parameters",
      +[](const decompression::VideoDecompressor & self) {
        return python::to_dict(self.parameters());
      },
      +[](decompression::VideoDecompressor & self, const bp::dict & dict) {
        self.parameters() = python::from_dict(dict);
      });

  bp::enum_<decompression::DecompressionType>("DecompressionType")
    .value("VIDEO", decompression::DecompressionType::VIDEO);

  bp::def(
    "create_decompressor",
    +[](const std::string & type) -> PythonDecompressorProxy * {
      return new PythonDecompressorProxy(decompression::create_decompressor(type));
    },
    bp::return_value_policy<bp::manage_new_object>());

  bp::def(
    "create_decompressor",
    +[](const decompression::DecompressionType & type) -> PythonDecompressorProxy * {
      return new PythonDecompressorProxy(decompression::create_decompressor(type));
    },
    bp::return_value_policy<bp::manage_new_object>());
}
