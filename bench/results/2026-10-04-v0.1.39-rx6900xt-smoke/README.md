# Strata v0.1.39 installation smoke check on RX 6900 XT

Checked on 2026-10-04 (Asia/Tokyo), on Windows with a Ryzen Threadripper 3990X
(64 cores, 128 logical processors), 127.9 GiB system RAM, and an RX 6900 XT
(gfx1030, 16 GiB). AMD driver: 32.0.21045.5002.

The source is upstream tag `v0.1.39`, commit
`6f32ec070f23ced9f50e704d854d775da52591ab`. The official Windows HIP asset was
downloaded from [the release](https://github.com/Niko1221/Strata/releases/tag/v0.1.39).
Its SHA-256 matched GitHub's asset digest:
`79672951142fdf9ac20268b5a57c6f902a74015cbbcd1a3a2b46d3791704741b`.
The installed `strata.exe` also matched the file inside that archive, SHA-256
`fe616b0f61b0ce1f18f37a9bd65542f883a64e78384230ffb42db927a59a6df2`.

The release includes the CPU affinity changes from
[PR #626](https://github.com/Niko1221/Strata/pull/626), which the maintainer took
as a squashed commit. This run does not remeasure that change's performance.

## Installation

The previous benchmark branch, `bench/rx6900xt-q2-long-context-20261004` at
`e3de3e7`, was clean and retained. A separate installation branch starts at the
release tag. The existing model configs, launchers and engine manifest were
backed up locally under `logs/update-v0.1.39-20261004/`.

Both Q2_0 and Coder configs previously pointed at an experimental affinity
engine. Their `exe` fields now point at the official `engine/strata.exe`; all
other config fields compare equal to the backups. Setup ran through
`setup.py --update --yes --prebuilt <directory-containing-the-verified-asset>`
using the existing virtual environment, and reported engine 0.1.39.

The four existing GGUF shards were reused. Their paths, sizes and modification
times compare equal before and after the update; no model download was needed.
This is a metadata preservation check, not a new full GGUF checksum run.

## Runtime checks

Q2_0 was started using the existing config on `127.0.0.1:8081`. The server
listens on loopback only, without an API key. Coder was updated but not loaded
for this check; both saved launchers use the same port.

| Check | Observed result |
| --- | --- |
| HIP device probe, with bundled ROCm libraries on PATH | RX 6900 XT, gfx1030, 16.0 GiB, wave32 |
| Engine startup log | `session is up (engine 0.1.39)`; 63 expert-pool workers plus the host thread |
| `GET /props` | `build_info: Strata 0.1.39`, one slot |
| `GET /health` | `status: ok`, `loaded: true`, model `qwen3.8-flash-next-q2_0`, context 131072 |
| `GET /v1/models` | Q2_0 listed as loaded, `n_ctx: 131072` |
| `POST /v1/chat/completions` | Requested `Strata update verified.` and received that exact text; finish reason `stop` |
| `POST /v1/responses` | Requested `Ready.` and received that exact text; status `completed` |
| `GET /` | HTTP 200 and the Strata HTML page |

Both generation checks used temperature 0, reasoning effort `none`, and a
32-token output limit. Q2_0 retains context 131072, KV `int8`, and resident KV
32768; Coder retains context 65536. Raw local setup and API evidence is in the
same ignored log directory as the backups.
The release hashes and API results are also saved in [verification.json](verification.json).

These are installation and short inference checks on the physical AMD GPU.
Long-context behavior, throughput changes, Coder inference, and interactive
browser behavior were not tested in this run.
