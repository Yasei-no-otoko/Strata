@echo off
rem Run a locally built HIP program with matching ROCm and the local VS OpenMP runtime.
rem This is a development launcher, not a redistributable ROCm/OpenMP package.
rem Example: run_windows_hip.bat dsv4_run.exe --model C:\models\model.gguf --selfcheck
setlocal EnableDelayedExpansion
if "%~1"=="" (echo Usage: run_windows_hip.bat program [arguments] & exit /b 2)
for %%I in ("%~dp0.") do set "DSV4_SRC=%%~fI"
if not defined ROCM_VENV set "ROCM_VENV=%DSV4_SRC%\..\.rocm-win"
if not defined BUILD_DIR set "BUILD_DIR=%DSV4_SRC%\build-hip-win"
if not exist "%ROCM_VENV%\Scripts\rocm-sdk.exe" (echo Set ROCM_VENV to the initialized ROCm venv. & exit /b 1)
for /f "delims=" %%R in ('"%ROCM_VENV%\Scripts\rocm-sdk.exe" path --root') do set "DSV4_ROCM=%%R"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "DSV4_VS="
if exist "%VSWHERE%" for /f "delims=" %%V in ('"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath') do set "DSV4_VS=%%V"
if defined DSV4_VS for /d %%R in ("%DSV4_VS%\VC\Redist\MSVC\*") do (
  if exist "%%R\debug_nonredist\x64\Microsoft.VC143.OpenMP.LLVM\libomp140.x86_64.dll" set "PATH=%%R\debug_nonredist\x64\Microsoft.VC143.OpenMP.LLVM;!PATH!"
)
set "PATH=%BUILD_DIR%;%DSV4_ROCM%\bin;%PATH%"
%*
exit /b %errorlevel%
