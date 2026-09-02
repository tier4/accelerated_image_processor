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

#pragma once

#include <accelerated_image_processor_common/datatype.hpp>
#include <accelerated_image_processor_common/parameter.hpp>
#include <accelerated_image_processor_common/processor.hpp>

#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/tuple.hpp>

#include <pyerrors.h>

#include <array>
#include <cstring>
#include <string>
#include <vector>

namespace accelerated_image_processor::python
{
namespace bp = boost::python;  // NOLINT

/**
 * @brief Convert std::vector<T> to python's list<T>.
 */
template <typename T>
bp::list vector_to_list(const std::vector<T> & vec)
{
  bp::list py_list;
  for (const auto & elem : vec) {
    py_list.append(elem);
  }
  return py_list;
}

/**
 * @brief Convert python's list<T> to std::vector<T>.
 */
template <typename T>
void list_to_vector(std::vector<T> & vec, const bp::object & iterable)
{
  bp::list py_list(iterable);
  const auto len = bp::len(py_list);
  vec.clear();
  vec.reserve(static_cast<size_t>(len));
  for (bp::ssize_t i = 0; i < len; ++i) {
    vec.emplace_back(bp::extract<T>(py_list[i]));
  }
}

/**
 * @brief Copy a contiguous one-byte Python buffer into a byte vector.
 *
 * NumPy arrays, memoryviews, bytes, and bytearrays take this bulk-copy path. Other
 * iterables retain the legacy element-by-element conversion for compatibility.
 */
inline void buffer_or_iterable_to_byte_vector(std::vector<uint8_t> & vec, const bp::object & object)
{
  Py_buffer view{};
  if (PyObject_GetBuffer(object.ptr(), &view, PyBUF_CONTIG_RO) == 0) {
    if (view.itemsize != 1) {
      PyBuffer_Release(&view);
      PyErr_SetString(PyExc_ValueError, "Image data buffer must have one-byte elements");
      bp::throw_error_already_set();
    }

    vec.resize(static_cast<std::size_t>(view.len));
    if (view.len > 0) {
      std::memcpy(vec.data(), view.buf, static_cast<std::size_t>(view.len));
    }
    PyBuffer_Release(&view);
    return;
  }

  // An unsupported-buffer error is expected for legacy list/tuple inputs.
  PyErr_Clear();
  list_to_vector<uint8_t>(vec, object);
}

/**
 * @brief Convert std::array<T, N> to python's list<T>.
 */
template <typename T, std::size_t N>
bp::list array_to_list(const std::array<T, N> & arr)
{
  bp::list py_list;
  for (const auto & elem : arr) {
    py_list.append(elem);
  }
  return py_list;
}

/**
 * @brief Convert python's list<T> to std::array<T, N>.
 */
template <typename T, std::size_t N>
void list_to_array(std::array<T, N> & arr, const bp::object & iterable)
{
  bp::list py_list(iterable);
  if (bp::len(py_list) != static_cast<bp::ssize_t>(N)) {
    PyErr_SetString(PyExc_ValueError, ("Expected list of length " + std::to_string(N)).c_str());
    bp::throw_error_already_set();
  }
  for (std::size_t i = 0; i < N; ++i) {
    arr[i] = bp::extract<T>(py_list[i]);
  }
}

/**
 * @brief Convert python's dict to common::ParameterMap.
 */
inline common::ParameterMap from_dict(const bp::dict & dict)
{
  common::ParameterMap map;

  for (bp::ssize_t i = 0; i < bp::len(dict); ++i) {
    bp::object key = dict.keys()[i];
    bp::object value = dict[key];

    std::string key_str = bp::extract<std::string>(bp::str(key));

    if (PyBool_Check(value.ptr())) {
      bool v = bp::extract<bool>(value);
      map[key_str] = v;
    } else if (PyLong_Check(value.ptr())) {
      long v = bp::extract<long>(value);  // NOLINT
      map[key_str] = static_cast<int>(v);
    } else if (PyFloat_Check(value.ptr())) {
      double v = bp::extract<double>(value);
      map[key_str] = v;
    } else {
      std::string v = bp::extract<std::string>(value);
      map[key_str] = v;
    }
  }
  return map;
}

/**
 * @brief Convert common::ParameterMap to python's dict.
 */
inline bp::dict to_dict(const common::ParameterMap & map)
{
  bp::dict dict;
  for (const auto & [key, value] : map) {
    const auto & name = key;
    std::visit(
      [&](const auto & v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, bool>) {
          dict[name] = v;
        } else if constexpr (std::is_same_v<T, int>) {
          dict[name] = v;
        } else if constexpr (std::is_same_v<T, double>) {
          dict[name] = v;
        } else if constexpr (std::is_same_v<T, std::string>) {
          dict[name] = v;
        }
      },
      value);
  }
  return dict;
}

/**
 * @brief Returns processed image or None. This is a wrapper for the process method.
 */
template <class ProcessorT>
inline bp::object process_or_none(ProcessorT * self, const common::Image & image)
{
  auto result = self->process(image);
  return result.has_value() ? bp::object(*result) : bp::object();
}

inline bp::object image_to_numpy(const bp::object & self, bool copy = false)
{
  const auto & image = bp::extract<const common::Image &>(self)();
  bp::object np = bp::import("numpy");
  bp::object frombuffer(np.attr("frombuffer"));
  bp::object uint8(np.attr("uint8"));
  bp::object array = frombuffer(self, uint8);

  if (image.format == common::ImageFormat::RAW) {
    const size_t expected_size =
      static_cast<size_t>(image.height) * static_cast<size_t>(image.step);
    const size_t actual_size = bp::extract<size_t>(array.attr("size"))();
    if (actual_size != expected_size) {
      PyErr_Format(
        PyExc_ValueError, "RAW image data has %zu bytes, expected %zu", actual_size, expected_size);
      bp::throw_error_already_set();
    }

    if (image.step == image.width * 3) {
      array = array.attr("reshape")(bp::make_tuple(image.height, image.width, 3));
    } else {
      array = array.attr("reshape")(bp::make_tuple(image.height, image.step));
    }
  }

  return copy ? array.attr("copy")() : array;
}
}  // namespace accelerated_image_processor::python
