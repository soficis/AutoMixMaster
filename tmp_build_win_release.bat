@echo off
setlocal
cd /d V:\AutoMixMaster
"C:\Program Files\CMake\bin\cmake.exe" -S . -B build_windows_x64_release -G "Visual Studio 18 2026" -A x64 -DBUILD_TESTING=OFF
if errorlevel 1 exit /b 1
"C:\Program Files\CMake\bin\cmake.exe" --build build_windows_x64_release --config Release -- /m
exit /b %errorlevel%
