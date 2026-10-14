# Strata 0.1.44.1 staged for Swift LAN on RX 6900 XT

On 2026-10-14 (Asia/Tokyo), the source was merged with the official
`v0.1.44.1` commit `296b59d2f73c8f04a805df29c2fa97655adace78`. The
Windows HIP archive's SHA-256 matched the release digest:
`e40bce7a2cfe47ea42d3f2af1e0adb92d14b8c02a29107c5a309b2cb518f5f52`.
Its `BUILD.json` identifies engine 0.1.44.1, ROCm `10.2.0a20260930`, and
`gfx1030`. The staged `strata-device.exe --list-devices` identified this PC's
RX 6900 XT as `gfx1030`, 16.0 GiB.

The new engine is in the ignored `engine-v0.1.44.1-hip` directory. The
existing `engine` directory and the original loopback Swift config were left
in place. The local GPU image encoder was copied into the new engine after
its SHA-256 (`0abe56ac988a727849602e652ed6501a4b023ba837bdfbcb3d8e705f692c29ed`)
and ROCm version were checked. The ignored
`strata-swift-huihui-iq3_xxs-v0.1.44.1.json` changes only the executable,
library, vision-executable and log paths. The Huihui Swift IQ3_XXS model,
pack, MTP, context and other settings remain those of the installed config.

`start-swift-rx6900xt-lan.bat` uses that config and the existing
`tools/start_swift_lan.ps1` launcher. It checks for the expected GPU and
engine version, and reads the bind address and API key from the ignored
`.env` file. The BAT enables `STRATA_HIP_PROMPT_F16=1`, an opt-in gfx1030
prompt speed path. Published RX 6900 XT measurements are on Linux and an
older engine; this Windows 0.1.44.1 combination has not been benchmarked.
FP16 prompt calculations can change generated tokens. The ordinary startup
and its loopback binding are unchanged.

`start-swift-rx6900xt-lan.bat -CheckOnly` passed without starting a server.
AMD setup, rerun-size, hotfix-tag, Windows AMD GPU, setup-choice and vision
elasticity tests passed (106 tests total). No model server was started or
restarted for this update, so there is no 0.1.44.1 inference, API, image or
physical LAN-client result yet. Run the BAT after any old service using port
8081 has been stopped by the user; it does not stop another process itself.
