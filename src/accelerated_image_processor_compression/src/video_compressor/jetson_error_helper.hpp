#pragma once
#include <cerrno>
#include <cstring>
#include <functional>
#include <iostream>

namespace accelerated_image_processor::compression
{
/**
 * @brief Holds the raw result of an NvEncoder/V4L2 call and readable message.
 */
struct EncStatus
{
  bool ok{true};        // true if NvEncoder returned 0
  std::string message;  // error text when !ok

  EncStatus() = default;
  EncStatus(bool r, const std::string & m = "") : ok(r), message(m) {}
};

/**
 * @brief Wraps a NvEncoder/V4L2 call that returns 0 on success and -1 on failure.
 *
 * The macro passes __FILE__ and __LINE__ automatically.
 */
inline EncStatus check_nvenc_call(int fn, const char * file, int line)
{
  int ret = fn;
  if (ret == 0) return EncStatus{true};
  std::string err = std::strerror(errno);
  std::string msg = std::string(file) + ":" + std::to_string(line) + " (" + err + ")";
  std::cerr << msg << std::endl;
  return EncStatus{false, msg};
}

struct EncResult
{
  bool ok{true};
  EncStatus status;

  EncResult() = default;
  explicit EncResult(const EncStatus & s) : ok(s.ok), status(s) {}
  explicit EncResult(bool o, const EncStatus & s) : ok(o), status(s) {}
  static EncResult success() { return EncResult(EncStatus{true, ""}); }
};

}  // namespace accelerated_image_processor::compression

/**
 * @brief Helper macro so the caller need only write: CHECK_NVENC(f, "Some operation")
 */
#define CHECK_NVENC(fn, msg)                                                              \
  {                                                                                       \
    auto _res =                                                                           \
      accelerated_image_processor::compression::check_nvenc_call(fn, __FILE__, __LINE__); \
    if (!_res.ok) {                                                                       \
      return accelerated_image_processor::compression::EncResult{_res};                   \
    }                                                                                     \
  }
