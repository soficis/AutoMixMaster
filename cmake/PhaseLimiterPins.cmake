# cmake/PhaseLimiterPins.cmake
# Single source of truth for PhaseLimiter release downloads and SHA-256 integrity pins.

set(AUTOMIX_PL_PIN_WINDOWS_X64_URL "https://github.com/ai-mastering/phaselimiter/releases/download/v0.2.0/phaselimiter-win.zip")
set(AUTOMIX_PL_PIN_WINDOWS_X64_SHA256 "cab2d30ad8d993a383749b30d9f6dc1911198d3aa309d5f631b70206e0162145")

set(AUTOMIX_PL_PIN_LINUX_X64_URL "https://github.com/soficis/phaselimiter/releases/download/v0.2.0-native2/phaselimiter-0.2.0-native2-linux-x64.tar.xz")
set(AUTOMIX_PL_PIN_LINUX_X64_SHA256 "994b587feee68b7ca3e95868ae8984df42806607a04acd6d4e854aaaa5389792")

set(AUTOMIX_PL_PIN_LINUX_ARM64_URL "https://github.com/soficis/phaselimiter/releases/download/v0.2.0-native2/phaselimiter-0.2.0-native2-linux-arm64.tar.xz")
set(AUTOMIX_PL_PIN_LINUX_ARM64_SHA256 "87bb4deb2df3588a13822020440a61159f944367a1d011a3e8f3421fc55a9000")

set(AUTOMIX_PL_PIN_MACOS_ARM64_URL "https://github.com/soficis/phaselimiter/releases/download/v0.2.0-native2/phaselimiter-0.2.0-native2-macos-arm64.tar.xz")
set(AUTOMIX_PL_PIN_MACOS_ARM64_SHA256 "6d7343359f4b38183b00151c3f56de62b113f7d6b18617cda1cc966f14b5aa8b")

set(AUTOMIX_PL_PIN_MACOS_X86_64_URL "https://github.com/soficis/phaselimiter/releases/download/v0.2.0-native2/phaselimiter-0.2.0-native2-macos-x86_64.tar.xz")
set(AUTOMIX_PL_PIN_MACOS_X86_64_SHA256 "3135a49e9bd711cab54c32e074c12fbbe62229479d868b2fe7bb8b3c58bbceb8")

function(automix_generate_phaselimiter_pins_header)
  set(_entries "")
  set(_platforms windows-x64 linux-x64 linux-arm64 macos-arm64 macos-x86_64)
  set(_var_windows-x64 WINDOWS_X64)
  set(_var_linux-x64 LINUX_X64)
  set(_var_linux-arm64 LINUX_ARM64)
  set(_var_macos-arm64 MACOS_ARM64)
  set(_var_macos-x86_64 MACOS_X86_64)

  foreach(_key IN LISTS _platforms)
    set(_var_key "${_var_${_key}}")
    set(_url "${AUTOMIX_PL_PIN_${_var_key}_URL}")
    set(_sha "${AUTOMIX_PL_PIN_${_var_key}_SHA256}")
    if(_url AND _sha)
      string(APPEND _entries "  {\"${_key}\", \"${_url}\", \"${_sha}\"},\n")
    endif()
  endforeach()

  string(REGEX REPLACE "\n$" "" AUTOMIX_PHASELIMITER_PIN_ROWS "${_entries}")

  configure_file(
    "${CMAKE_SOURCE_DIR}/src/renderers/PhaseLimiterPins.h.in"
    "${CMAKE_BINARY_DIR}/generated/renderers/PhaseLimiterPins.h"
    @ONLY
  )
endfunction()
