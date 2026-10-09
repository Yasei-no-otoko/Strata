# Strata 0.1.41 update and Swift startup

On 2026-10-09 (Asia/Tokyo), the existing Windows installation was updated
from Strata 0.1.40 to the latest official release, 0.1.41. The existing
Huihui Swift 1.5 IQ3_XXS model is running at `http://127.0.0.1:8081` on a
Threadripper 3990X, 128 GB RAM and Radeon RX 6900 XT (gfx1030, 16 GiB).

Release source `fb58e0dbc8399662c0e47c76578c6e878b14f6cf` was merged into
the previous installation branch. The existing Windows HIP vision changes
were retained; the CMake conflict resolution preserves the upstream
`STRATA_PORTABLE=ON` default and the optional HIP backend.

The official `strata-windows-x64-hip.zip` SHA-256 was checked against the
GitHub release API before running:

```powershell
.\.venv\Scripts\python.exe setup.py --update --yes --prebuilt C:\Dev\Strata\logs\update-v0.1.41-20261009
```

The archive digest is
`e8701d1d183f0f84742f6b9dcde7c5589cfd86271cdec70646f88cf441ee6716`.
Setup warns that it cannot verify a local archive itself; the independent
hash check passed. The installed `strata.exe` also matches the executable
inside that verified archive.

The Swift config is byte-identical, including the model paths, loopback
binding, port 8081, context 131072 and image settings. All 68 existing files
under `C:\Dev\Strata-data` retained their paths, sizes and modification
times. No model download was needed. A full engine backup, config, launcher,
update output and raw verification records are kept locally under
`logs/update-v0.1.41-20261009/`.

The previously built HIP image encoder is unchanged. It uses the same
ROCm version as the new official engine, `10.2.0a20260930`; its separate
vision metadata was preserved in `engine/BUILD.json`. It is a retained
local encoder, not a new encoder shipped by the official 0.1.41 package.

| Check | Result |
| --- | --- |
| HIP device detection and device selftest | RX 6900 XT; passed |
| Runtime log | HIP runtime, RX 6900 XT, engine 0.1.41 |
| `/health`, `/props`, `/v1/models`, web root | Loaded Swift, 131072 context, images enabled, Strata 0.1.41, HTTP 200 |
| Chat Completions | Exact `Swift ready.` |
| Responses API | Exact `Strata updated.` |
| Image Chat Completions | Correct red circle, blue square and `HIP2026` |
| Image encoder runtime | ROCm backend on RX 6900 XT; successful warm-up |
| Existing setup update tests | 8 passed |
| Python dependency check | No broken requirements |
| Final health | Loaded and healthy |

The image request used the existing fixture from
`../2026-10-07-windows-hip-vision-rx6900xt/vision-chat.png`, with a prompt
that did not supply the expected image contents. These are short functional
checks, not a throughput or long-context benchmark. The retained image
encoder was not rebuilt in this update. The running server was started
as a hidden background process; the existing `run-swift-huihui-iq3_xxs.bat`
can start it again after the process stops.

See [verification.json](verification.json) for hashes, responses and
runtime evidence.
