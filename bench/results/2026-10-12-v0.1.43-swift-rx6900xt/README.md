# Strata 0.1.43 update and Swift restart

On 2026-10-12 (Asia/Tokyo), this Windows installation was updated from
Strata 0.1.41 to the official 0.1.43 release. The existing Huihui Swift 1.5
IQ3_XXS model was restarted on the Radeon RX 6900 XT (gfx1030, 16 GiB).
The current Private Ethernet address is `192.168.0.3`; the authenticated
web UI is at `http://192.168.0.3:8081/` and the API base is
`http://192.168.0.3:8081/v1`. Loopback `127.0.0.1:8081` also works with
the same API key. The previous `192.168.0.4` address is no longer assigned
to this PC.

The source was merged with upstream commit
`7695859aa13afb248bdb4d36f9b2e95707db079d`, retaining the local
Windows HIP vision change and `tools/start_swift_lan.ps1`. The official
`strata-windows-x64-hip.zip` archive was downloaded and checked against
the release API digest before being given to setup:

```powershell
.\.venv\Scripts\python.exe setup.py --update --yes --prebuilt C:\Dev\Strata\logs\update-v0.1.43-20261012
```

The verified archive SHA-256 is
`128435c77990f466b0339dd4611cbb1703e2950ee01ded029538eafbb5eefa27`.
Setup warns that a local `--prebuilt` archive has no built-in release hash
check. The separate check above passed, and the installed engine matches
the engine inside the archive.

The Swift configuration is byte-identical to its backup. All 68 files in
`C:\Dev\Strata-data` retained their paths, sizes and modification times;
no model download was needed. The pre-update engine and launch files are
backed up in the ignored `logs/update-v0.1.43-20261012/` directory.
The official HIP archive has no image encoder. The existing locally built
HIP encoder was retained unchanged, uses the same ROCm version as the new
engine (`10.2.0a20260930`), and passed a real image request after restart.

The existing ignored `.env` API key was reused without placing its value
in the command line, logs or this report. The base Swift config remains
loopback-only; LAN binding is enabled by the separate launcher.

| Check | Result |
| --- | --- |
| HIP device detection and selftest | RX 6900 XT, gfx1030; passed |
| Setup update and AMD setup tests | 8 and 36 passed |
| `/health` via LAN and loopback | Swift loaded, 131072 context, images enabled, API key present |
| `/props` and `/v1/status` | Strata/engine 0.1.43 |
| `/v1/models` with no key or wrong key | HTTP 401 |
| `/v1/models` with correct key via LAN and loopback | HTTP 200 |
| Web UI via LAN address | HTTP 200 |
| Chat Completions and Responses API | Exact `Swift ready.` and `Strata updated.` |
| Image Chat Completions | Red circle, blue square and `HIP2026` recognized |
| Runtime logs | ROCm vision on RX 6900 XT; HIP engine 0.1.43 |

The requests were sent from this PC to its LAN address. Reachability and
inference from a separate physical device were not tested. The short
requests here are functional checks, not performance measurements.
See [verification.json](verification.json) for the recorded hashes and
responses; it contains no API key.
