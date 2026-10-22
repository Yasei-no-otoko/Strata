# Strata v0.2.2 update — RX 6900 XT

**Status: completed.** The signed Windows HIP engine update and RX 6900 XT GPU vision verification passed. The GPU reader workaround is a separate local executable; the official engine package remains intact.

## Updated installation and preservation

Updated official Strata source to `v0.2.2` (`011c9e600a93bbc9fd451c563501b5831c32bbca`) and engine to `0.1.47`, Windows HIP RDNA2 (`gfx1030`/`gfx1031`). Host: Windows 11, AMD Threadripper 3990X, 128 GiB RAM, Radeon RX 6900 XT (16 GiB, `gfx1030`), driver `32.0.21045.16010`. Swift Huihui `IQ3_XXS` remains at context 131072, cap 300, and 64 threads.

The official engine tar SHA-256 `f9e8954f2bf1c153994441ffd69006f0492e1d8300650098058ede1a4d7c742a` matched the GitHub asset digest and `SHA256SUMS`. The signed feed verified the package signature and all 320 files. Post-install, all 320 matched the official payload after normalizing `BUILD.json`'s `source: prebuilt` field. Engine component build ID is `76f90e58436d`, ROCm `10.2.0a20260930`, signed. The official payload is preserved unchanged.

The model, pack, and MTP inventory is unchanged: 60 files, 172,250,859,305 bytes, with matching relative paths, sizes, and modification times. Backups are in `logs/update-v0.2.2-20261022/engine-v0.1.44.1/` and `logs/update-v0.2.2-20261022/configs/`. `engine/.previous` retains engine `0.1.44.1`.

The normal Swift config changed only `vision.exe` and `vision.cpu_exe`: the primary reader points to `engine/strata-vision-hip-gfx1030.exe`, and CPU fallback points to the official `engine/strata-vision.exe`. All other config fields are unchanged, including GPU mode, model source, context, cap, threads, and engine arguments. The normal run BAT and LAN wrapper are unchanged. The LAN BAT targets the standard config and expects engine `0.1.47`. `.env` ACL is unchanged; the key was not printed.

## Engine and GPU vision verification

The official `strata-vision-hip.exe` reported architectures `gfx1200,gfx1201` and rejected the RX 6900 XT. A separately built and named local reader fixes this compatibility gap without replacing any official package file. It reports `gfx1030`; 275/275 build and link steps succeeded. `engine/LOCAL_VISION_BUILD.json` records the source version and commit, llama.cpp pin, and build metadata; vision source was unmodified.

The build used the v0.2.2 vision source, llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d`, ROCm `10.2.0a20260930`, Visual Studio 2022, CMake/Ninja, `STRATA_VISION_HIP_ARCHS=gfx1030`, `STRATA_PORTABLE=ON` (AVX2 baseline), `GGML_HIP_NO_VMM=ON`, CUDA/SYCL off, and 8 parallel jobs. The exact reproducible invocation is recorded in `logs/update-v0.2.2-20261022/build-vision-gfx1030.bat`; build output is in `logs/update-v0.2.2-20261022/vision-build.log`.

The signed-feed updater's full owner-attested one-token test-start passed before the local vision reader was installed. Its first attempt through the Windows venv redirector did not accept readiness because the redirector PID (`70876`) differed from the actual server PID (`48772`); that temporary server stopped cleanly. Retrying with base Python and `PYTHONPATH=.venv/Lib/site-packages` passed the package checks and test-start (`generation_ready: true`, one-token response, then stopped). The temporary verification harness required a private attestation environment setting; this was handled locally without a product source change.

Final API checks used the local `gfx1030` reader in an isolated temporary server. Health reported loaded model, images enabled, and API key enabled. Text chat returned `Swift ready.` (0.547 s); Responses returned `Strata updated.` (0.438 s). Two fixture images were correctly identified as a red circle and blue square, with text `HIP2026` and `HIP2027` (4.328 s and 3.844 s). Authentication returned HTTP 401 without a key and HTTP 200 with the configured key. The HIP reader warmed up on 289 image tokens in 1.1 s; the engine identified the RX 6900 XT (`gfx1030`); no CPU fallback occurred. The temporary server stopped successfully and no Strata process remained.

Detailed results are in `verification.json`. Supporting records are in `logs/update-v0.2.2-20261022/`, including `runtime-verification.json`, `runtime-server.log`, `runtime-engine.log`, `local-vision-build.json`, `vision-build.log`, and `device-selftest.log`.

No physical LAN client was tested, and no performance claim is made.
