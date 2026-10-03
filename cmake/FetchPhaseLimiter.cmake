# cmake/FetchPhaseLimiter.cmake
# Fetches pinned PhaseLimiter release archives and stages assets beside executables.

include(FetchContent)

function(automix_phaselimiter_platform_key OUT_VAR)
  set(_os "${CMAKE_SYSTEM_NAME}")
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _proc)

  if(_os STREQUAL "Darwin" OR APPLE)
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
      message(FATAL_ERROR "Unsupported macOS processor: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()

  elseif(_os STREQUAL "Linux")
    if(_proc MATCHES "arm64|aarch64")
      set(${OUT_VAR} "linux-arm64" PARENT_SCOPE)
      return()
    elseif(_proc MATCHES "x86_64|amd64")
      set(${OUT_VAR} "linux-x64" PARENT_SCOPE)
      return()
    else()
      message(FATAL_ERROR "Unsupported Linux processor: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()

  elseif(_os STREQUAL "Windows" OR WIN32)
    if(_proc MATCHES "x86_64|amd64")
      set(${OUT_VAR} "windows-x64" PARENT_SCOPE)
      return()
    else()
      message(FATAL_ERROR "Unsupported Windows processor: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()

  else()
    message(FATAL_ERROR "Unsupported platform for PhaseLimiter: ${CMAKE_SYSTEM_NAME}")
  endif()
endfunction()

if(CMAKE_SCRIPT_MODE_FILE)
  return()
endif()

set(AUTOMIX_PHASELIMITER_ROOT "")

if(WIN32)
  if(EXISTS "${CMAKE_SOURCE_DIR}/assets/phaselimiter/bin/phase_limiter.exe")
    set(AUTOMIX_PHASELIMITER_ROOT "${CMAKE_SOURCE_DIR}/assets/phaselimiter")
  else()
    set(_pl_url "https://github.com/ai-mastering/phaselimiter/releases/download/v0.2.0/phaselimiter-win.zip")
    set(_pl_hash "SHA256=cab2d30ad8d993a383749b30d9f6dc1911198d3aa309d5f631b70206e0162145")
  endif()
elseif(APPLE)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64")
    set(_pl_url "https://github.com/soficis/phaselimiter/releases/download/v0.2.0-native1/phaselimiter-0.2.0-macos-arm64.tar.xz")
    set(_pl_hash "SHA256=bffd93614efc9f7d74b3ac148eef3731339dabbdaffaf9ab8912fc562473b722")
  endif()
elseif(UNIX)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    set(_pl_url "https://github.com/ai-mastering/phaselimiter/releases/download/v0.2.0/release.tar.xz")
    set(_pl_hash "SHA256=0b382ba78b030926f706345d1b00d5f45890ba384c8a4e960320e3ee992bd163")
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
