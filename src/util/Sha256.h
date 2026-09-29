#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

// Single-file SHA-256 (FIPS 180-4). This replaces FNV-1a-64 as the model-pack
// trust anchor: FNV is explicitly non-cryptographic, so a pack manifest could be
// edited to match a tampered model file. The download hubs already vendored this
// algorithm; it is promoted here so there is exactly one implementation.
namespace automix::util {

namespace sha256_detail {

inline constexpr uint32_t kRoundConstants[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

inline uint32_t rotr(const uint32_t x, const int n) { return (x >> n) | (x << (32 - n)); }

inline void processBlock(uint32_t state[8], const uint8_t block[64]) {
  uint32_t w[64];
  for (int i = 0; i < 16; ++i) {
    w[i] = (static_cast<uint32_t>(block[i * 4]) << 24) |
           (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
           (static_cast<uint32_t>(block[i * 4 + 2]) << 8) |
           static_cast<uint32_t>(block[i * 4 + 3]);
  }
  for (int i = 16; i < 64; ++i) {
    const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
    const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }

  uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
  uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
  for (int i = 0; i < 64; ++i) {
    const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
    const uint32_t ch = (e & f) ^ (~e & g);
    const uint32_t temp1 = h + S1 + ch + kRoundConstants[i] + w[i];
    const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
    const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t temp2 = S0 + maj;
    h = g; g = f; f = e; e = d + temp1;
    d = c; c = b; b = a; a = temp1 + temp2;
  }

  state[0] += a; state[1] += b; state[2] += c; state[3] += d;
  state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

} // namespace sha256_detail

// Lowercase 64-hex digest, or an empty string when the file cannot be opened.
inline std::string fileSha256(const std::filesystem::path& filePath) {
  std::ifstream file(filePath, std::ios::binary);
  if (!file.is_open()) {
    return {};
  }

  uint32_t state[8] = {
      0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  };

  uint8_t block[64];
  uint64_t totalBytes = 0;
  while (file) {
    file.read(reinterpret_cast<char*>(block), 64);
    const auto bytesRead = static_cast<size_t>(file.gcount());
    totalBytes += bytesRead;
    if (bytesRead == 64) {
      sha256_detail::processBlock(state, block);
    } else {
      block[bytesRead] = 0x80;
      std::fill(block + bytesRead + 1, block + 64, static_cast<uint8_t>(0));
      if (bytesRead >= 56) {
        sha256_detail::processBlock(state, block);
        std::fill(block, block + 64, static_cast<uint8_t>(0));
      }
      const uint64_t totalBits = totalBytes * 8;
      for (int i = 0; i < 8; ++i) {
        block[63 - i] = static_cast<uint8_t>(totalBits >> (i * 8));
      }
      sha256_detail::processBlock(state, block);
    }
  }

  static constexpr char digits[] = "0123456789abcdef";
  std::string hex;
  hex.reserve(64);
  for (int i = 0; i < 8; ++i) {
    for (int j = 3; j >= 0; --j) {
      const uint8_t byte = static_cast<uint8_t>(state[i] >> (j * 8));
      hex.push_back(digits[(byte >> 4) & 0x0f]);
      hex.push_back(digits[byte & 0x0f]);
    }
  }
  return hex;
}

inline bool isSha256Hex(const std::string_view candidate) {
  if (candidate.size() != 64) {
    return false;
  }
  for (const auto c : candidate) {
    const bool hex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
    if (!hex) {
      return false;
    }
  }
  return true;
}

// An empty `expectedSha256` means there is nothing to check and returns true. A
// file that cannot be hashed is a failure, so callers fail closed by default.
inline bool verifyFileSha256(const std::filesystem::path& filePath,
                             const std::string_view expectedSha256,
                             std::string* detail) {
  if (expectedSha256.empty()) {
    return true;
  }

  const auto computed = fileSha256(filePath);
  if (computed.empty()) {
    if (detail != nullptr) {
      *detail = "Unable to compute SHA-256 for: " + filePath.string();
    }
    return false;
  }

  if (computed.size() != expectedSha256.size()) {
    if (detail != nullptr) {
      *detail = "SHA-256 length mismatch for " + filePath.filename().string();
    }
    return false;
  }

  for (size_t i = 0; i < computed.size(); ++i) {
    const auto lhs = static_cast<char>(std::tolower(static_cast<unsigned char>(computed[i])));
    const auto rhs = static_cast<char>(std::tolower(static_cast<unsigned char>(expectedSha256[i])));
    if (lhs != rhs) {
      if (detail != nullptr) {
        *detail = "SHA-256 mismatch for " + filePath.filename().string();
      }
      return false;
    }
  }

  return true;
}

} // namespace automix::util
