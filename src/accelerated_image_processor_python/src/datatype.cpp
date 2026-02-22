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

#include <accelerated_image_processor_common/datatype.hpp>

#include <boost/python.hpp>
#include <boost/python/errors.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <array>
#include <vector>

namespace py = boost::python;                 // NOLINT
using namespace accelerated_image_processor;  // NOLINT

namespace
{
template <typename T>
py::list vector_to_list(const std::vector<T> & vec)
{
  py::list py_list;
  for (const auto & elem : vec) {
    py_list.append(elem);
  }
  return py_list;
}

template <typename T>
void list_to_vector(std::vector<T> & vec, const py::object & iterable)
{
  py::list py_list(iterable);
  for (py::ssize_t i = 0; i < py::len(py_list); ++i) {
    vec.push_back(py::extract<T>(py_list[i]));
  }
}

template <typename T, std::size_t N>
py::list array_to_list(const std::array<T, N> & arr)
{
  py::list py_list;
  for (const auto & elem : arr) {
    py_list.append(elem);
  }
  return py_list;
}

template <typename T, std::size_t N>
void list_to_array(std::array<T, N> & arr, const py::object & iterable)
{
  py::list py_list(iterable);
  if (py::len(py_list) != static_cast<py::ssize_t>(N)) {
    PyErr_SetString(PyExc_ValueError, ("Expected list of length " + std::to_string(N)).c_str());
    py::throw_error_already_set();
  }
  for (std::size_t i = 0; i < N; ++i) {
    arr[i] = py::extract<T>(py_list[i]);
  }
}
}  // namespace

BOOST_PYTHON_MODULE(accelerated_image_processor_python_datatype)
{
  // ------- Enums -------
  py::enum_<common::ImageEncoding>("ImageEncoding")
    .value("RGB", common::ImageEncoding::RGB)
    .value("BGR", common::ImageEncoding::BGR);

  py::enum_<common::ImageFormat>("ImageFormat")
    .value("RAW", common::ImageFormat::RAW)
    .value("JPEG", common::ImageFormat::JPEG)
    .value("PNG", common::ImageFormat::PNG);

  py::enum_<common::DistortionModel>("DistortionModel")
    .value("PLUMB_BOB", common::DistortionModel::PLUMB_BOB)
    .value("RATIONAL_POLYNOMIAL", common::DistortionModel::RATIONAL_POLYNOMIAL)
    .value("EQUIDISTANT", common::DistortionModel::EQUIDISTANT);

  // ------- Image -------
  py::class_<common::Image>("Image")
    .def_readwrite("frame_id", &common::Image::frame_id)
    .def_readwrite("timestamp", &common::Image::timestamp)
    .def_readwrite("height", &common::Image::height)
    .def_readwrite("width", &common::Image::width)
    .def_readwrite("step", &common::Image::step)
    .def_readwrite("encoding", &common::Image::encoding)
    .def_readwrite("format", &common::Image::format)
    .add_property(
      "data",  // [uint8_t; height * width * step]
      +[](const common::Image & img) { return vector_to_list<uint8_t>(img.data); },
      +[](common::Image & img, const py::object & iterable) {
        list_to_vector<uint8_t>(img.data, iterable);
      })
    .def("is_valid", &common::Image::is_valid);

  // ------- CameraInfo -------
  py::class_<common::CameraInfo>("CameraInfo")
    .def_readwrite("frame_id", &common::CameraInfo::frame_id)
    .def_readwrite("timestamp", &common::CameraInfo::timestamp)
    .def_readwrite("height", &common::CameraInfo::height)
    .def_readwrite("width", &common::CameraInfo::width)
    .def_readwrite("distortion_model", &common::CameraInfo::distortion_model)
    .add_property(
      "d",  // [double; N]
      +[](const common::CameraInfo & info) { return vector_to_list<double>(info.d); },
      +[](common::CameraInfo & info, const py::object & iterable) {
        list_to_vector<double>(info.d, iterable);
      })
    .add_property(
      "k",  // [double; 9]
      +[](const common::CameraInfo & info) { return array_to_list<double, 9>(info.k); },
      +[](common::CameraInfo & info, const py::object & iterable) {
        list_to_array<double, 9>(info.k, iterable);
      })
    .add_property(
      "r",  // [double, 9]
      +[](const common::CameraInfo & info) { return array_to_list<double, 9>(info.r); },
      +[](common::CameraInfo & info, const py::object & iterable) {
        list_to_array<double, 9>(info.r, iterable);
      })
    .add_property(
      "p",  // [double, 12]
      +[](const common::CameraInfo & info) { return array_to_list<double, 12>(info.p); },
      +[](common::CameraInfo & info, const py::object & iterable) {
        list_to_array<double, 12>(info.p, iterable);
      })
    .def_readwrite("binning_x", &common::CameraInfo::binning_x)
    .def_readwrite("binning_y", &common::CameraInfo::binning_y)
    .def_readwrite("roi", &common::CameraInfo::roi);

  // ------- Roi -------
  py::class_<common::Roi>("Roi")
    .def_readwrite("x_offset", &common::Roi::x_offset)
    .def_readwrite("y_offset", &common::Roi::y_offset)
    .def_readwrite("width", &common::Roi::width)
    .def_readwrite("height", &common::Roi::height)
    .def_readwrite("do_rectify", &common::Roi::do_rectify);
}
