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

#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

namespace bp = boost::python;                 // NOLINT
using namespace accelerated_image_processor;  // NOLINT

namespace
{
/**
 * @brief Python-facing proxy for compression::Compressor.
 *
 * This proxy is required because Jetson video encoders emit encoded frames asynchronously via
 * postprocess callbacks, while Python users expect Compressor.process() to synchronously return an
 * Image (or None on failure).
 *
 * Keeping this adaptation in the Python binding layer preserves the original C++ compressor
 * behavior and timing semantics used by existing C++ tests, while still providing a convenient
 * Python API that can wait briefly and return the callback-produced frame.
 */
class PythonCompressorProxy
{
public:
  explicit PythonCompressorProxy(std::unique_ptr<compression::Compressor> compressor)
  : compressor_(std::move(compressor))
  {
    if (compressor_) {
      compressor_
        ->register_postprocess<PythonCompressorProxy, &PythonCompressorProxy::on_postprocess>(this);
      if (compressor_->backend() == compression::CompressorBackend::JETSON) {
        await_async_result_ = true;
      }
    }
  }

  std::optional<common::Image> process(const common::Image & image)
  {
    if (!compressor_) {
      return std::nullopt;
    }

    if (auto result = compressor_->process(image); result.has_value()) {
      return result;
    }

    if (!await_async_result_) {
      return std::nullopt;
    }

    std::unique_lock<std::mutex> lock(result_queue_mutex_);
    if (!result_queue_cv_.wait_for(
          lock, std::chrono::milliseconds(500), [this]() { return !result_queue_.empty(); })) {
      return std::nullopt;
    }

    auto result = std::move(result_queue_.front());
    result_queue_.pop_front();
    return result;
  }

  compression::CompressorBackend backend() const { return compressor_->backend(); }

  common::ParameterMap & parameters() { return compressor_->parameters(); }
  const common::ParameterMap & parameters() const { return compressor_->parameters(); }

  void register_postprocess(const bp::object & callback)
  {
    if (callback.is_none()) {
      callback_ = bp::object();
      callback_enabled_ = false;
      return;
    }

    if (!PyCallable_Check(callback.ptr())) {
      PyErr_SetString(PyExc_TypeError, "callback must be callable");
      bp::throw_error_already_set();
    }

    callback_ = callback;
    callback_enabled_ = true;
  }

private:
  void on_postprocess(const common::Image & image)
  {
    if (callback_enabled_) {
      PyGILState_STATE gil_state = PyGILState_Ensure();
      try {
        callback_(image);
      } catch (const bp::error_already_set &) {
        PyErr_Print();
      }
      PyGILState_Release(gil_state);
    }

    {
      std::lock_guard<std::mutex> lock(result_queue_mutex_);
      result_queue_.push_back(image);
    }
    result_queue_cv_.notify_one();
  }

  std::unique_ptr<compression::Compressor> compressor_;
  bool await_async_result_{false};
  bool callback_enabled_{false};
  bp::object callback_;

  std::mutex result_queue_mutex_;
  std::condition_variable result_queue_cv_;
  std::deque<common::Image> result_queue_;
};
}  // namespace

BOOST_PYTHON_MODULE(accelerated_image_processor_python_compression)
{
  // -----------
  // Compressor
  // -----------

  // NOTE: Bind PythonCompressorProxy as an abstract base class which is named Compressor and cannot
  // be instantiated directly.
  bp::class_<PythonCompressorProxy, boost::noncopyable>("Compressor", bp::no_init)
    .def("process", &python::process_or_none<PythonCompressorProxy>)
    .def("register_postprocess", &PythonCompressorProxy::register_postprocess)
    .add_property("backend", &PythonCompressorProxy::backend)
    .add_property(
      "parameters",
      +[](const PythonCompressorProxy & self) { return python::to_dict(self.parameters()); },
      +[](PythonCompressorProxy & self, const bp::dict & dict) {
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
    +[](const std::string & type) -> PythonCompressorProxy * {
      return new PythonCompressorProxy(compression::create_compressor(type));
    },
    bp::return_value_policy<bp::manage_new_object>());

  bp::def(
    "create_compressor",
    +[](const compression::CompressionType & type) -> PythonCompressorProxy * {
      return new PythonCompressorProxy(compression::create_compressor(type));
    },
    bp::return_value_policy<bp::manage_new_object>());
}
