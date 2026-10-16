# Strata 0.1.44.1 installed in the normal Windows HIP engine path

On 2026-10-17 (Asia/Tokyo), GitHub still listed `v0.1.44.1` as the latest
Strata release. This checkout already included its release commit
`296b59d2f73c8f04a805df29c2fa97655adace78`, and the RX 6900 XT LAN
launcher already pointed to a staged 0.1.44.1 engine. The normal `engine`
directory still contained 0.1.43, so this operation installed the staged
0.1.44.1 HIP engine there as well.

The official Windows HIP archive was rechecked against its published SHA-256:
`e40bce7a2cfe47ea42d3f2af1e0adb92d14b8c02a29107c5a309b2cb518f5f52`.
The staged engine declares `gfx1030` and ROCm `10.2.0a20260930`. Its local
HIP image encoder had the expected SHA-256
`0abe56ac988a727849602e652ed6501a4b023ba837bdfbcb3d8e705f692c29ed`.
The active `engine/strata.exe` now matches the staged engine's SHA-256:
`c4709f17ef974ffa34af4f0c357160a89249cabf6e9a698fbe6d0dc65a3243fa`.

Before replacement, port 8081 had no listener, no Strata engine/server
process was found, and the old executable allowed an exclusive open. The old
engine and a copy of the Swift config are retained in the ignored
`logs/update-v0.1.44.1-20261017/` directory. The original config's SHA-256
remained `3213f66bd950da1376ceb56af91c37be95f402b1f228d124c0da9c6ed22292c8`.
The 69 files in `C:\Dev\Strata-data` were not modified by the engine swap;
the file count was 69 before and after. No model download occurred.

The installed `strata-device.exe --list-devices` found the RX 6900 XT as
`gfx1030`, 16.0 GiB, and `--selftest` passed. Both the normal-config and
RX 6900 XT LAN launcher `-CheckOnly` checks passed for engine 0.1.44.1.

A service was observed running during final verification; this update did
not start, stop or restart it. Its engine and vision processes were using
the separate `engine-v0.1.44.1-hip` directory, and its log reported
`session is up (engine 0.1.44.1)`. `/health` reported the Swift model loaded
with images and an API key. `/props` reported `Strata 0.1.44.1`. From this
same PC to its LAN address, `/v1/models` returned 401 without a key and 200
with the `.env` key. The key value was not recorded. No text/image inference
or separate physical LAN client was tested in this operation.
