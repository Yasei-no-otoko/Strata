@echo off
setlocal
rem RX 6900 XT (gfx1030): opt in to the faster FP16 prompt GEMM path.
rem This can change generated tokens; the default Strata startup is unchanged.
rem The LAN bind address and API key come from the local .env file.
set "STRATA_HIP_PROMPT_F16=1"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\start_swift_lan.ps1" -ConfigName "strata-swift-huihui-iq3_xxs-v0.1.44.1.json" -ExpectedGpu "AMD Radeon RX 6900 XT" -ExpectedEngineVersion "0.1.44.1" %*
exit /b %ERRORLEVEL%
