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
#include <atomic>
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
    // Release the GIL while decoding so that multiple decompressors can run in parallel
    // and other Python threads (e.g., ROS executors) are not starved.
    GilRelease release;
    return decompressor_->process(image);
  }

  void register_postprocess(const bp::object & callback)
  {
    // Treat None as disabling the postprocess callback.
    // Publish the disabled flag before clearing the callable. on_postprocess() can run
    // without the GIL and must not invoke a callback that is already cleared.
    if (callback.is_none()) {
      callback_enabled_.store(false, std::memory_order_release);
      callback_ = bp::object();
      return;
    }

    // Ensure that the provided object is callable
    if (!PyCallable_Check(callback.ptr())) {
      PyErr_SetString(PyExc_TypeError, "register_postprocess expects a callable or None");
      bp::throw_error_already_set();
    }
    callback_ = callback;
    callback_enabled_.store(true, std::memory_order_release);
  }

  common::ParameterMap & parameters() { return decompressor_->parameters(); }
  const common::ParameterMap & parameters() const { return decompressor_->parameters(); }

private:
  class GilRelease
  {
  public:
    GilRelease() : state_(PyEval_SaveThread()) {}
    ~GilRelease() { PyEval_RestoreThread(state_); }
    GilRelease(const GilRelease &) = delete;
    GilRelease & operator=(const GilRelease &) = delete;

  private:
    PyThreadState * state_;
  };

  class GilAcquire
  {
  public:
    GilAcquire() : state_(PyGILState_Ensure()) {}
    ~GilAcquire() { PyGILState_Release(state_); }
    GilAcquire(const GilAcquire &) = delete;
    GilAcquire & operator=(const GilAcquire &) = delete;

  private:
    PyGILState_STATE state_;
  };

  // Called from process() while the GIL is released.
  void on_postprocess(const common::Image & image)
  {
    if (!callback_enabled_.load(std::memory_order_acquire)) {
      return;
    }

    GilAcquire acquire;
    if (callback_enabled_.load(std::memory_order_acquire)) {
      callback_(image);
    }
  }

  std::unique_ptr<decompression::Decompressor> decompressor_;
  std::atomic<bool> callback_enabled_{false};
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
      +[](const PythonDecompressorProxy & self) { return python::to_dict(self.parameters()); },
      +[](PythonDecompressorProxy & self, const bp::dict & dict) {
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
