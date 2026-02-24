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

#include "binding.hpp"

#include <accelerated_image_processor_compression/builder.hpp>
#include <accelerated_image_processor_compression/compressor.hpp>

#include <boost/python.hpp>

#include <string>

namespace bp = boost::python;                 // NOLINT
using namespace accelerated_image_processor;  // NOLINT

BOOST_PYTHON_MODULE(accelerated_image_processor_python_compression)
{
  // -----------
  // Compressor
  // -----------

  // NOTE: Bind Compressor as an abstract base class, which cannot be instantiated directly.
  bp::class_<compression::Compressor, boost::noncopyable>("Compressor", bp::no_init)
    .def("process", &python::process_or_none<compression::Compressor>)
    .def_readonly("backend", &compression::Compressor::backend)
    .add_property(
      "parameters",
      +[](const compression::Compressor & self) { return python::to_dict(self.parameters()); },
      +[](compression::Compressor & self, const bp::dict & dict) {
        self.parameters() = python::from_dict(dict);
      });

  bp::enum_<compression::CompressorBackend>("CompressionBackend")
    .value("JETSON", compression::CompressorBackend::JETSON)
    .value("NVJPEG", compression::CompressorBackend::NVJPEG)
    .value("CPU", compression::CompressorBackend::CPU);

  bp::enum_<compression::CompressionType>("CompressionType")
    .value("JPEG", compression::CompressionType::JPEG)
    .value("H264", compression::CompressionType::H264)
    .value("H265", compression::CompressionType::H265)
    .value("AV1", compression::CompressionType::AV1);

  bp::def(
    "create_compressor",
    +[](const std::string & type) -> compression::Compressor * {
      return compression::create_compressor(type).release();
    },
    bp::return_value_policy<bp::manage_new_object>());

  bp::def(
    "create_compressor",
    +[](const compression::CompressionType & type) -> compression::Compressor * {
      return compression::create_compressor(type).release();
    },
    bp::return_value_policy<bp::manage_new_object>());
}
