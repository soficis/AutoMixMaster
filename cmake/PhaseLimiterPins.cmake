# cmake/PhaseLimiterPins.cmake
# Single source of truth for PhaseLimiter release downloads and SHA-256 integrity pins.

set(AUTOMIX_PL_PIN_WINDOWS_X64_URL "https://github.com/ai-mastering/phaselimiter/releases/download/v0.2.0/phaselimiter-win.zip")
set(AUTOMIX_PL_PIN_WINDOWS_X64_SHA256 "cab2d30ad8d993a383749b30d9f6dc1911198d3aa309d5f631b70206e0162145")

set(AUTOMIX_PL_PIN_LINUX_X64_URL "https://github.com/ai-mastering/phaselimiter/releases/download/v0.2.0/release.tar.xz")
set(AUTOMIX_PL_PIN_LINUX_X64_SHA256 "0b382ba78b030926f706345d1b00d5f45890ba384c8a4e960320e3ee992bd163")

set(AUTOMIX_PL_PIN_MACOS_ARM64_URL "https://github.com/soficis/phaselimiter/releases/download/v0.2.0-native1/phaselimiter-0.2.0-macos-arm64.tar.xz")
set(AUTOMIX_PL_PIN_MACOS_ARM64_SHA256 "bffd93614efc9f7d74b3ac148eef3731339dabbdaffaf9ab8912fc562473b722")

set(AUTOMIX_PL_PIN_MACOS_X86_64_URL "")
set(AUTOMIX_PL_PIN_MACOS_X86_64_SHA256 "")

set(AUTOMIX_PL_PIN_LINUX_ARM64_URL "")
set(AUTOMIX_PL_PIN_LINUX_ARM64_SHA256 "")

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
