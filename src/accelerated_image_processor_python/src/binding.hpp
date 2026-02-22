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

#include <array>
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
  for (bp::ssize_t i = 0; i < bp::len(py_list); ++i) {
    vec.push_back(bp::extract<T>(py_list[i]));
  }
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
inline bp::object process_or_none(common::BaseProcessor * self, const common::Image & image)
{
  auto result = self->process(image);
  return result.has_value() ? bp::object(*result) : bp::object();
}
}  // namespace accelerated_image_processor::python
