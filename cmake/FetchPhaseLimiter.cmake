# cmake/FetchPhaseLimiter.cmake
# Fetches pinned PhaseLimiter release archives and stages assets beside executables.

include(FetchContent)

function(automix_phaselimiter_platform_key OUT_VAR)
  set(_os "${CMAKE_SYSTEM_NAME}")
  if(NOT _os)
    if(APPLE)
      set(_os "Darwin")
    elseif(WIN32)
      set(_os "Windows")
    else()
      set(_os "${CMAKE_HOST_SYSTEM_NAME}")
    endif()
  endif()

  set(_proc_raw "${CMAKE_SYSTEM_PROCESSOR}")
  if(NOT _proc_raw)
    set(_proc_raw "${CMAKE_HOST_SYSTEM_PROCESSOR}")
  endif()
  if(NOT _proc_raw AND WIN32)
    set(_proc_raw "$ENV{PROCESSOR_ARCHITECTURE}")
  endif()
  if(NOT _proc_raw AND (UNIX OR APPLE))
    execute_process(COMMAND uname -m OUTPUT_VARIABLE _uname_m OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    set(_proc_raw "${_uname_m}")
  endif()
  string(TOLOWER "${_proc_raw}" _proc)

  if(_os STREQUAL "Darwin")
    if(CMAKE_OSX_ARCHITECTURES)
      list(LENGTH CMAKE_OSX_ARCHITECTURES _num_archs)
      if(_num_archs GREATER 1)
        message(FATAL_ERROR "PhaseLimiter does not support universal macOS architectures: ${CMAKE_OSX_ARCHITECTURES}")
      endif()
      string(TOLOWER "${CMAKE_OSX_ARCHITECTURES}" _osx_arch)
      if(_osx_arch STREQUAL "arm64" OR _osx_arch STREQUAL "aarch64")
        set(${OUT_VAR} "macos-arm64" PARENT_SCOPE)
        return()
      elseif(_osx_arch STREQUAL "x86_64" OR _osx_arch STREQUAL "amd64")
        set(${OUT_VAR} "macos-x86_64" PARENT_SCOPE)
        return()
      else()
        message(FATAL_ERROR "Unsupported macOS architecture in CMAKE_OSX_ARCHITECTURES: ${CMAKE_OSX_ARCHITECTURES}")
      endif()
    endif()

    if(_proc MATCHES "arm64|aarch64")
      set(${OUT_VAR} "macos-arm64" PARENT_SCOPE)
      return()
    elseif(_proc MATCHES "x86_64|amd64")
      set(${OUT_VAR} "macos-x86_64" PARENT_SCOPE)
      return()
    else()
      message(FATAL_ERROR "Unsupported macOS processor: ${_proc_raw}")
    endif()

  elseif(_os STREQUAL "Linux")
    if(_proc MATCHES "arm64|aarch64")
      set(${OUT_VAR} "linux-arm64" PARENT_SCOPE)
      return()
    elseif(_proc MATCHES "x86_64|amd64")
      set(${OUT_VAR} "linux-x64" PARENT_SCOPE)
      return()
    else()
      message(FATAL_ERROR "Unsupported Linux processor: ${_proc_raw}")
    endif()

  elseif(_os STREQUAL "Windows")
    if(_proc MATCHES "x86_64|amd64|arm64|aarch64")
      set(${OUT_VAR} "windows-x64" PARENT_SCOPE)
      return()
    else()
      message(FATAL_ERROR "Unsupported Windows processor: ${_proc_raw}")
    endif()

  else()
    message(FATAL_ERROR "Unsupported platform for PhaseLimiter: ${_os}")
  endif()
endfunction()

if(CMAKE_SCRIPT_MODE_FILE OR NOT AUTOMIX_FETCH_PHASELIMITER)
  return()
endif()

set(AUTOMIX_PHASELIMITER_ROOT "")

include("${CMAKE_CURRENT_LIST_DIR}/PhaseLimiterPins.cmake")
automix_phaselimiter_platform_key(_platform_key)

set(_pl_url "")
set(_pl_hash "")

if(_platform_key STREQUAL "windows-x64")
  if(EXISTS "${CMAKE_SOURCE_DIR}/assets/phaselimiter/bin/phase_limiter.exe")
    set(AUTOMIX_PHASELIMITER_ROOT "${CMAKE_SOURCE_DIR}/assets/phaselimiter")
  else()
    set(_pl_url "${AUTOMIX_PL_PIN_WINDOWS_X64_URL}")
    if(AUTOMIX_PL_PIN_WINDOWS_X64_SHA256)
      set(_pl_hash "SHA256=${AUTOMIX_PL_PIN_WINDOWS_X64_SHA256}")
    endif()
  endif()
elseif(_platform_key STREQUAL "linux-x64")
  set(_pl_url "${AUTOMIX_PL_PIN_LINUX_X64_URL}")
  if(AUTOMIX_PL_PIN_LINUX_X64_SHA256)
    set(_pl_hash "SHA256=${AUTOMIX_PL_PIN_LINUX_X64_SHA256}")
  endif()
elseif(_platform_key STREQUAL "linux-arm64")
  set(_pl_url "${AUTOMIX_PL_PIN_LINUX_ARM64_URL}")
  if(AUTOMIX_PL_PIN_LINUX_ARM64_SHA256)
    set(_pl_hash "SHA256=${AUTOMIX_PL_PIN_LINUX_ARM64_SHA256}")
  endif()
elseif(_platform_key STREQUAL "macos-arm64")
  set(_pl_url "${AUTOMIX_PL_PIN_MACOS_ARM64_URL}")
  if(AUTOMIX_PL_PIN_MACOS_ARM64_SHA256)
    set(_pl_hash "SHA256=${AUTOMIX_PL_PIN_MACOS_ARM64_SHA256}")
  endif()
elseif(_platform_key STREQUAL "macos-x86_64")
  set(_pl_url "${AUTOMIX_PL_PIN_MACOS_X86_64_URL}")
  if(AUTOMIX_PL_PIN_MACOS_X86_64_SHA256)
    set(_pl_hash "SHA256=${AUTOMIX_PL_PIN_MACOS_X86_64_SHA256}")
  endif()
endif()

if(NOT AUTOMIX_PHASELIMITER_ROOT AND _pl_url)
  message(STATUS "Fetching PhaseLimiter archive: ${_pl_url}")
  FetchContent_Declare(
    phaselimiter_fetched
    URL "${_pl_url}"
    URL_HASH "${_pl_hash}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(phaselimiter_fetched)

  if(EXISTS "${phaselimiter_fetched_SOURCE_DIR}/bin")
    set(AUTOMIX_PHASELIMITER_ROOT "${phaselimiter_fetched_SOURCE_DIR}")
  elseif(EXISTS "${phaselimiter_fetched_SOURCE_DIR}/phaselimiter/bin")
    set(AUTOMIX_PHASELIMITER_ROOT "${phaselimiter_fetched_SOURCE_DIR}/phaselimiter")
  else()
    file(GLOB _pl_subdirs LIST_DIRECTORIES true "${phaselimiter_fetched_SOURCE_DIR}/*")
    foreach(_sub IN LISTS _pl_subdirs)
      if(IS_DIRECTORY "${_sub}" AND EXISTS "${_sub}/bin")
        set(AUTOMIX_PHASELIMITER_ROOT "${_sub}")
        break()
      endif()
    endforeach()
  endif()
endif()

if(UNIX AND AUTOMIX_PHASELIMITER_ROOT)
  file(GLOB _pl_bins "${AUTOMIX_PHASELIMITER_ROOT}/bin/*")
  foreach(_bin IN LISTS _pl_bins)
    if(NOT IS_DIRECTORY "${_bin}")
      file(CHMOD "${_bin}"
        PERMISSIONS
          OWNER_READ OWNER_WRITE OWNER_EXECUTE
          GROUP_READ GROUP_EXECUTE
          WORLD_READ WORLD_EXECUTE)
    endif()
  endforeach()
endif()

function(automix_stage_phaselimiter TARGET_NAME)
  if(NOT AUTOMIX_PHASELIMITER_ROOT)
    return()
  endif()

  set(_staging_commands)
  foreach(_dir bin resource licenses)
    if(EXISTS "${AUTOMIX_PHASELIMITER_ROOT}/${_dir}")
      list(APPEND _staging_commands COMMAND ${CMAKE_COMMAND} -E copy_directory
           "${AUTOMIX_PHASELIMITER_ROOT}/${_dir}"
           "$<TARGET_FILE_DIR:${TARGET_NAME}>/assets/phaselimiter/${_dir}")
    endif()
  endforeach()
  foreach(_file LICENSE README.md)
    if(EXISTS "${AUTOMIX_PHASELIMITER_ROOT}/${_file}")
      list(APPEND _staging_commands COMMAND ${CMAKE_COMMAND} -E copy_if_different
           "${AUTOMIX_PHASELIMITER_ROOT}/${_file}"
           "$<TARGET_FILE_DIR:${TARGET_NAME}>/assets/phaselimiter/${_file}")
    endif()
  endforeach()

  if(_staging_commands)
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${TARGET_NAME}>/assets/phaselimiter"
      ${_staging_commands}
      COMMENT "Staging PhaseLimiter assets next to ${TARGET_NAME}"
      VERBATIM
    )
  endif()
endfunction()
