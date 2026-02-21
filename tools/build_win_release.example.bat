@echo off
setlocal

rem Example Windows Release build script for AutoMixMaster.
rem Copy this file to the repository root and adjust as needed before use.
rem
rem Optionally set CMAKE_BIN to override the CMake executable path, e.g.:
rem   set CMAKE_BIN=C:\Program Files\CMake\bin\cmake.exe
rem
rem Optionally set BUILD_DIR to override the build directory, e.g.:
rem   set BUILD_DIR=build_windows_x64_release
rem
rem Optionally set GENERATOR to override the CMake generator, e.g.:
rem   set GENERATOR=Visual Studio 18 2026

if not defined CMAKE_BIN set CMAKE_BIN=cmake
if not defined BUILD_DIR set BUILD_DIR=build_windows_x64_release
if not defined GENERATOR set GENERATOR=Visual Studio 18 2026

cd /d "%~dp0.."
"%CMAKE_BIN%" -S . -B "%BUILD_DIR%" -G "%GENERATOR%" -A x64 -DBUILD_TESTING=OFF
if errorlevel 1 exit /b 1
"%CMAKE_BIN%" --build "%BUILD_DIR%" --config Release -- /m
exit /b %errorlevel%
