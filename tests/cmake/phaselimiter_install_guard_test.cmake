# tests/cmake/phaselimiter_install_guard_test.cmake

cmake_minimum_required(VERSION 3.20)

get_filename_component(_root_dir "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
include("${_root_dir}/cmake/FetchPhaseLimiter.cmake")
automix_phaselimiter_platform_key(_host_key)

set(_temp_dir "${CMAKE_CURRENT_BINARY_DIR}/install_guard_scratch")
file(REMOVE_RECURSE "${_temp_dir}")
file(MAKE_DIRECTORY "${_temp_dir}/install_prefix")

# Verify that if binary or resource is missing, the guard fails with non-zero exit and expected error
set(_dummy_cmake_install "${_temp_dir}/test_guard.cmake")
file(WRITE "${_dummy_cmake_install}" "
set(CMAKE_INSTALL_PREFIX \"${_temp_dir}/install_prefix\")
set(_automix_pl_dest \"assets/phaselimiter\")
set(_automix_native_bin_name \"phase_limiter\")
if(\"${_host_key}\" STREQUAL \"windows-x64\")
  set(_automix_native_bin_name \"phase_limiter.exe\")
endif()
set(_pl_bin_check \"\${CMAKE_INSTALL_PREFIX}/\${_automix_pl_dest}/bin/\${_automix_native_bin_name}\")
set(_pl_res_check \"\${CMAKE_INSTALL_PREFIX}/\${_automix_pl_dest}/resource/mastering_reference.json\")
if(NOT EXISTS \"\${_pl_bin_check}\" OR NOT EXISTS \"\${_pl_res_check}\")
  message(FATAL_ERROR \"Package has no native PhaseLimiter for ${_host_key}\")
endif()
")

execute_process(
  COMMAND "${CMAKE_COMMAND}" -P "${_dummy_cmake_install}"
  RESULT_VARIABLE _res
  ERROR_VARIABLE _err
  OUTPUT_VARIABLE _out
)

if(_res EQUAL 0)
  message(FATAL_ERROR "Expected guard script to fail on missing binary, but it succeeded!")
endif()

set(_full_err "${_out}${_err}")
if(NOT _full_err MATCHES "Package has no native PhaseLimiter for ${_host_key}")
  message(FATAL_ERROR "Expected error message to contain 'Package has no native PhaseLimiter for ${_host_key}', got:\n${_full_err}")
endif()

# Verify that when binary and resource exist, the guard succeeds
file(MAKE_DIRECTORY "${_temp_dir}/install_prefix/assets/phaselimiter/bin")
file(MAKE_DIRECTORY "${_temp_dir}/install_prefix/assets/phaselimiter/resource")
set(_bin_ext "")
if("${_host_key}" STREQUAL "windows-x64")
  set(_bin_ext ".exe")
endif()
file(WRITE "${_temp_dir}/install_prefix/assets/phaselimiter/bin/phase_limiter${_bin_ext}" "dummy binary")
file(WRITE "${_temp_dir}/install_prefix/assets/phaselimiter/resource/mastering_reference.json" "{}")

execute_process(
  COMMAND "${CMAKE_COMMAND}" -P "${_dummy_cmake_install}"
  RESULT_VARIABLE _res2
  ERROR_VARIABLE _err2
  OUTPUT_VARIABLE _out2
)

if(NOT _res2 EQUAL 0)
  message(FATAL_ERROR "Guard script failed when files were present:\n${_out2}${_err2}")
endif()

file(REMOVE_RECURSE "${_temp_dir}")
message(STATUS "cmake_phaselimiter_install_guard test passed.")
