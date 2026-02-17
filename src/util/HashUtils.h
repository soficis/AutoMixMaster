#pragma once

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>

namespace automix::util {

constexpr uint64_t kFnv1a64OffsetBasis = 14695981039346656037ull;
constexpr uint64_t kFnv1a64Prime = 1099511628211ull;

inline uint64_t fnv1a64Update(uint64_t hash, const void* data, const size_t size) noexcept {
  if (data == nullptr || size == 0) {
    return hash;
  }

  const auto* bytes = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= kFnv1a64Prime;
  }
  return hash;
}

inline uint64_t fnv1a64(const void* data, const size_t size) noexcept {
  return fnv1a64Update(kFnv1a64OffsetBasis, data, size);
}

inline uint64_t fnv1a64(const std::string_view input) noexcept {
  return fnv1a64(input.data(), input.size());
}

inline std::string toHex(const uint64_t value) {
  std::ostringstream out;
  out << std::hex << value;
  return out.str();
}

} // namespace automix::util
