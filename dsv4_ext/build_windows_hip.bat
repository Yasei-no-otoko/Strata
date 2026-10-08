@echo off
rem Build only the experimental DeepSeek extension; the installed Strata engine is not replaced.
rem Requires Visual Studio C++ Build Tools, CMake, Ninja, and an initialized TheRock ROCm venv.
rem Set ROCM_VENV to an existing Strata .rocm-win, and DSV4_HIP_ARCHS to the target GPU.
setlocal EnableDelayedExpansion
for %%I in ("%~dp0.") do set "DSV4_SRC=%%~fI"
if not defined ROCM_VENV set "ROCM_VENV=%DSV4_SRC%\..\.rocm-win"
if not defined BUILD_DIR set "BUILD_DIR=%DSV4_SRC%\build-hip-win"
if not defined BUILD_JOBS set "BUILD_JOBS=4"
if not defined DSV4_HIP_ARCHS (
  echo Set DSV4_HIP_ARCHS to the GPU architecture, for example gfx1030 for RX 6900 XT.
  exit /b 1
)
if not exist "%ROCM_VENV%\Scripts\rocm-sdk.exe" (
  echo An initialized ROCm venv is required. Set ROCM_VENV to the Strata .rocm-win directory.
  echo See docs/AMD_HIP.md for the Windows ROCm build setup.
  exit /b 1
)
for /f "delims=" %%R in ('"%ROCM_VENV%\Scripts\rocm-sdk.exe" path --root') do set "DSV4_ROCM=%%R"
if not exist "%DSV4_ROCM%\lib\llvm\bin\clang++.exe" (echo ROCm compiler not found. & exit /b 1)
set "DSV4_ROCM_F=%DSV4_ROCM:\=/%"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "DSV4_VS="
if exist "%VSWHERE%" for /f "delims=" %%V in ('"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath') do set "DSV4_VS=%%V"
if not defined DSV4_VS (echo Visual Studio C++ Build Tools are required. & exit /b 1)
call "%DSV4_VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set "HIP_PLATFORM=amd"
set "HIP_PATH=%DSV4_ROCM%"
set "ROCM_PATH=%DSV4_ROCM%"
set "PATH=%DSV4_ROCM%\bin;%DSV4_ROCM%\lib\llvm\bin;%PATH%"
cmake -G Ninja -S "%DSV4_SRC%" -B "%BUILD_DIR%" -DCMAKE_BUILD_TYPE=Release ^
  -DDSV4_WITH_HIP=ON -DDSV4_WITH_CUDA=OFF "-DCMAKE_HIP_ARCHITECTURES=%DSV4_HIP_ARCHS%" ^
  "-DCMAKE_CXX_COMPILER=%DSV4_ROCM_F%/lib/llvm/bin/clang++.exe" ^
  "-DCMAKE_HIP_COMPILER=%DSV4_ROCM_F%/lib/llvm/bin/clang++.exe" ^
  "-DCMAKE_HIP_COMPILER_ROCM_ROOT=%DSV4_ROCM_F%" "-DCMAKE_PREFIX_PATH=%DSV4_ROCM_F%" ^
  "-DCMAKE_HIP_FLAGS=--rocm-path=%DSV4_ROCM_F% --rocm-device-lib-path=%DSV4_ROCM_F%/lib/llvm/amdgcn/bitcode" || exit /b 1
cmake --build "%BUILD_DIR%" --parallel %BUILD_JOBS% || exit /b 1
rem Prefer the matching HIP DLLs beside the executable over any driver-installed System32 copy.
for %%D in (amdhip64_7.dll amdhip64.dll amd_comgr.dll) do (
  if exist "%DSV4_ROCM%\bin\%%D" copy /y "%DSV4_ROCM%\bin\%%D" "%BUILD_DIR%\" >nul
)
echo Built %BUILD_DIR%\dsv4_run.exe
echo Run ctest --test-dir "%BUILD_DIR%" --output-on-failure with the ROCm bin directory on PATH.
endlocal
