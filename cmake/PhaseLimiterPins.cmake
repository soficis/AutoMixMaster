# cmake/PhaseLimiterPins.cmake
# Single source of truth for PhaseLimiter release downloads and SHA-256 integrity pins.

set(AUTOMIX_PL_PIN_WINDOWS_X64_URL "https://github.com/ai-mastering/phaselimiter/releases/download/v0.2.0/phaselimiter-win.zip")
set(AUTOMIX_PL_PIN_WINDOWS_X64_SHA256 "cab2d30ad8d993a383749b30d9f6dc1911198d3aa309d5f631b70206e0162145")

set(AUTOMIX_PL_PIN_LINUX_X64_URL "https://github.com/soficis/phaselimiter/releases/download/v0.2.0-native3/phaselimiter-0.2.0-native3-linux-x64.tar.xz")
set(AUTOMIX_PL_PIN_LINUX_X64_SHA256 "8ba28b31f823555c3c368a6d3b4757c1980c981b6abe726e80d724f050aeefaa")

set(AUTOMIX_PL_PIN_LINUX_ARM64_URL "https://github.com/soficis/phaselimiter/releases/download/v0.2.0-native3/phaselimiter-0.2.0-native3-linux-arm64.tar.xz")
set(AUTOMIX_PL_PIN_LINUX_ARM64_SHA256 "714867eaf3ca9c33f68648ae3d073323b0990d816cca7b767ded41456461441c")

set(AUTOMIX_PL_PIN_MACOS_ARM64_URL "https://github.com/soficis/phaselimiter/releases/download/v0.2.0-native3/phaselimiter-0.2.0-native3-macos-arm64.tar.xz")
set(AUTOMIX_PL_PIN_MACOS_ARM64_SHA256 "42614e95f2d185bf40498a6e1da199f547a2c7bd427e74ed74ee35c9d1512a43")

set(AUTOMIX_PL_PIN_MACOS_X86_64_URL "https://github.com/soficis/phaselimiter/releases/download/v0.2.0-native3/phaselimiter-0.2.0-native3-macos-x86_64.tar.xz")
set(AUTOMIX_PL_PIN_MACOS_X86_64_SHA256 "e63e2b755fdaa80c104aa6ff5a0866e0f3754fbb79ed84efd696af583201afbe")

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
