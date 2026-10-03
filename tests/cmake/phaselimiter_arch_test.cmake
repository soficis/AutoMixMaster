# tests/cmake/phaselimiter_arch_test.cmake

cmake_minimum_required(VERSION 3.20)

get_filename_component(_root_dir "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
include("${_root_dir}/cmake/FetchPhaseLimiter.cmake")

if(TEST_SUBCASE STREQUAL "universal")
  set(CMAKE_SYSTEM_NAME "Darwin")
  set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64")
  set(CMAKE_SYSTEM_PROCESSOR "arm64")
  automix_phaselimiter_platform_key(_key)
  message(FATAL_ERROR "Should not reach here: universal arch did not trigger error")
endif()

# Test 1: Darwin, no CMAKE_OSX_ARCHITECTURES, arm64 processor
set(CMAKE_SYSTEM_NAME "Darwin")
set(CMAKE_OSX_ARCHITECTURES "")
set(CMAKE_SYSTEM_PROCESSOR "arm64")
automix_phaselimiter_platform_key(_key)
if(NOT _key STREQUAL "macos-arm64")
  message(FATAL_ERROR "Test 1 failed: expected macos-arm64, got '${_key}'")
endif()

# Test 2: Darwin, CMAKE_OSX_ARCHITECTURES=x86_64
set(CMAKE_SYSTEM_NAME "Darwin")
set(CMAKE_OSX_ARCHITECTURES "x86_64")
set(CMAKE_SYSTEM_PROCESSOR "arm64")
automix_phaselimiter_platform_key(_key)
if(NOT _key STREQUAL "macos-x86_64")
  message(FATAL_ERROR "Test 2 failed: expected macos-x86_64, got '${_key}'")
endif()

# Test 3: Darwin, CMAKE_OSX_ARCHITECTURES=arm64 with processor x86_64 (explicit arch wins)
set(CMAKE_SYSTEM_NAME "Darwin")
set(CMAKE_OSX_ARCHITECTURES "arm64")
set(CMAKE_SYSTEM_PROCESSOR "x86_64")
automix_phaselimiter_platform_key(_key)
if(NOT _key STREQUAL "macos-arm64")
  message(FATAL_ERROR "Test 3 failed: expected macos-arm64, got '${_key}'")
endif()

# Test 4: Linux, processor aarch64
set(CMAKE_SYSTEM_NAME "Linux")
set(CMAKE_OSX_ARCHITECTURES "")
set(CMAKE_SYSTEM_PROCESSOR "aarch64")
automix_phaselimiter_platform_key(_key)
if(NOT _key STREQUAL "linux-arm64")
  message(FATAL_ERROR "Test 4 failed: expected linux-arm64, got '${_key}'")
endif()

# Test 5: Linux, processor AMD64 (case-insensitive)
set(CMAKE_SYSTEM_NAME "Linux")
set(CMAKE_OSX_ARCHITECTURES "")
set(CMAKE_SYSTEM_PROCESSOR "AMD64")
automix_phaselimiter_platform_key(_key)
if(NOT _key STREQUAL "linux-x64")
  message(FATAL_ERROR "Test 5 failed: expected linux-x64, got '${_key}'")
endif()

# Test 6: Windows, processor AMD64
set(CMAKE_SYSTEM_NAME "Windows")
set(CMAKE_OSX_ARCHITECTURES "")
set(CMAKE_SYSTEM_PROCESSOR "AMD64")
automix_phaselimiter_platform_key(_key)
if(NOT _key STREQUAL "windows-x64")
  message(FATAL_ERROR "Test 6 failed: expected windows-x64, got '${_key}'")
endif()

# Test 7: Darwin, CMAKE_OSX_ARCHITECTURES="arm64;x86_64" -> FATAL_ERROR containing "universal"
execute_process(
  COMMAND "${CMAKE_COMMAND}" -DTEST_SUBCASE=universal -P "${CMAKE_CURRENT_LIST_FILE}"
  RESULT_VARIABLE _proc_res
  OUTPUT_VARIABLE _proc_out
  ERROR_VARIABLE _proc_err
)

if(_proc_res EQUAL 0)
  message(FATAL_ERROR "Test 7 failed: expected universal architecture to trigger non-zero exit code")
endif()

set(_all_proc_output "${_proc_out}${_proc_err}")
string(TOLOWER "${_all_proc_output}" _lower_err)
if(NOT _lower_err MATCHES "universal")
  message(FATAL_ERROR "Test 7 failed: expected 'universal' in error output, got:\n${_all_proc_output}")
endif()

message(STATUS "All automix_phaselimiter_platform_key tests passed successfully.")
