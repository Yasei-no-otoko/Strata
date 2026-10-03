# Q2_0 long-prompt benchmark — RX 6900 XT / Threadripper 3990X

**Completed:** Q2_0 read 32,768-token prompts at **319.0 tokens/s** and generated at **53.6 tokens/s**; at exactly 128,000 input tokens, the medians were **311.6** and **49.9 tokens/s**. Each result is the median of three fresh-prompt runs. [Japanese results](RESULTS-JA.md).

## Scope

This measures Strata's original Qwen3.8-Flash-Next GSQ-RCO **Q2_0** model on one Windows AMD GPU at two long prompt lengths. It is a single-machine result. The synthetic code-review prompts follow the repository benchmark harness style and are token-counted with the exact OpenAI chat renderer; they are not byte-for-byte the same prompt strings used in the published RTX 5070 matrix.

The executable is a locally built, processor-group-aware binary, **not an official release binary**. It comes from source commit [`814554564dec0cdd4000a5994ead0f5851ae2fde`](https://github.com/Niko1221/Strata/pull/626), and its SHA-256 is recorded in [engine-build.json](engine-build.json) and [runtime-hashes.json](runtime-hashes.json). The source change addresses Windows processor-group affinity on high-core-count CPUs. Results describe this executable and configuration only.

## Hardware and software

- Windows 11 Pro, build 10.0.26300.
- AMD Ryzen Threadripper 3990X: 64 physical cores, 128 logical processors, across two Windows processor groups. This Zen 2 CPU has no AVX-512; the native model pack therefore uses AVX2 expert kernels.
- AMD Radeon RX 6900 XT, 16 GiB, gfx1030; AMD driver 32.0.21045.5002; bundled HIP/ROCm runtime 10.2.0a20260930.
- 128 GiB installed RAM (127.876 GiB visible to Windows); 82.613 GiB available before model load. Eight 16 GiB DDR4-3200 modules were reported.
- 2 TB NVMe SSD. The desktop was interactive and other applications were left running.
- Strata source/server commit `b4a9b941e1506e031ea197d39c410d9e14c3e791`; engine version 0.1.38. Build and executable details are in [hardware.json](hardware.json) and [engine-build.json](engine-build.json).

## Model and runtime configuration

Model source: [`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`.

The two Q2_0 GGUF shards were checked against the source revision's published LFS SHA-256 values; both matched. The verified filenames, sizes and hashes are in [model-source.json](model-source.json) and [model-integrity.json](model-integrity.json). Strata prepared its native AVX2 pack from shard 1; the pack uses the matching tokenizer and template. PLE is read from shard 2 on the NVMe SSD.

Measured engine settings:

- One RX 6900 XT through HIP; no images and no calibration.
- Context capacity: 131,072 tokens for both workloads.
- INT8 KV cache, with 32,768 cells per attention layer resident in VRAM and the remaining KV stored in pinned system RAM.
- Automatic prefill; automatic expert cache. Startup selected 4,354 expert slots (5.61 GiB VRAM) after reserving 700 MiB and accounting for the MTP draft head. The default expert profile was prefilled without eviction.
- MTP enabled with `--spec 4 --spec-min-p 0.5`; automatic worker selection chose 63 expert-pool workers plus the host thread.
- No gfx1030 hipBLASLt tuning table was available for the bundled library, so dense prompt operations used plain hipBLAS.

The exact engine arguments are preserved in [measured-config.json](measured-config.json); startup details are in [startup-engine.txt](startup-engine.txt). The original installer configuration used a 132,096-token capacity, but the measured server was restarted with the 131,072-token capacity above; [ready-status.json](ready-status.json) confirms the measured context.

## Method

One warm-up request with a 256-token output cap was excluded. The measured run order is three fresh prompts at 32,768 tokens, followed by three at 128,000 tokens. The benchmark uses a unique nonce at the start of each user message and the local tokenizer plus OpenAI renderer to target the prompt lengths. It records actual API usage and engine history for every request; the table below uses the verified actual counts. Each measured request uses temperature 0, reasoning disabled and a 256-token output cap. At the 128,000-token target, 256 output tokens and the server's 8-token context margin total 128,264 tokens, within the measured 131,072-token capacity.

The model stays loaded for the whole suite. Model loading and startup expert-cache population are excluded; the warm-up occurs before the measured requests. The expert cache and operating-system file cache remain warm between measured requests, with no server restart or explicit file-cache flush. Prompt-prefix reuse is checked from both API usage and engine history for each run; all six measured requests reused zero prompt tokens.

Client TTFT is measured from request dispatch to the first non-empty streamed text delta. Total latency ends when the SSE stream completes. Prompt throughput is freshly read prompt tokens divided by the engine's prompt-read time; decode throughput uses engine-generated tokens divided by engine decode time. These are separate from client latency. Runs that fail, are cancelled, reuse prompt tokens, or generate fewer than the cap will be identified in the results rather than silently folded into the successful fresh-prompt summary.

Reproduction uses the recorded [model revision](model-source.json), [measured engine configuration](measured-config.json), and locally built engine hash in [engine-build.json](engine-build.json). With the server already running on `127.0.0.1:8081`, run this from `C:\Dev\Strata` in PowerShell (choose a new output directory for a new run):

```powershell
python .\bench\results\2026-10-04-rx6900xt-q2_0\benchmark.py `
  --root C:\Dev\Strata `
  --pack C:\Dev\Strata-data\packs\q2_0 `
  --url http://127.0.0.1:8081 `
  --out C:\Dev\Strata\bench\results\2026-10-04-rx6900xt-q2_0\reproduction `
  --log C:\Dev\Strata\strata-q2_0.log `
  --targets 32768,128000 `
  --runs 3
```

This creates a separate output set; it does not overwrite the measured run directory. The raw request bodies, output text, SSE usage/timings, before/after metrics, and engine-log snippets accompany the summary.

## Results

Measured on 2026-10-04, Asia/Tokyo. All six measured requests completed with exactly the requested input length, **zero reused tokens**, and **256 generated tokens** (`finish_reason=length`). One separate 256-token warm-up is excluded.

Median of three runs; parentheses are minimum–maximum. Throughput is tokens/s; client latency is seconds.

| Actual input tokens | Prompt tokens/s | Decode tokens/s | First text (s) | Total (s) |
| ---: | ---: | ---: | ---: | ---: |
| 32,768 | 319.0 (315.7–320.6) | 53.6 (52.7–53.7) | 102.90 (102.49–104.03) | 107.73 (107.23–108.76) |
| 128,000 | 311.6 (311.5–313.9) | 49.9 (48.4–52.3) | 411.43 (408.51–411.52) | 416.30 (413.77–416.62) |

Individual trials:

| Input tokens | Run | Prompt tokens/s | Decode tokens/s | First text (s) | Total (s) | MTP drafts accepted / offered |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32,768 | 1 | 315.70 | 53.73 | 104.026 | 108.761 | 166 / 215 |
| 32,768 | 2 | 319.00 | 52.72 | 102.898 | 107.727 | 166 / 229 |
| 32,768 | 3 | 320.55 | 53.63 | 102.486 | 107.232 | 164 / 215 |
| 128,000 | 1 | 313.88 | 48.43 | 408.510 | 413.767 | 155 / 225 |
| 128,000 | 2 | 311.48 | 49.91 | 411.521 | 416.622 | 158 / 206 |
| 128,000 | 3 | 311.60 | 52.27 | 411.430 | 416.300 | 163 / 209 |

The 128K workload is exactly **128,000 tokens**, matching the requested convention rather than 131,072. Both workloads used the same 131,072-token capacity and the same loaded server session.

[summary.json](measured/summary.json) contains the full statistics; [results.csv](results.csv) contains individual trial values. The [measured directory](measured/) preserves every full request, output text, raw SSE stream, before/after metrics, and appended engine-log lines. [Full session log](session-engine.txt).

The offline [verification](measured/verification.json) passed for all six trials: each started with no request in flight, introduced exactly one engine-history entry, and matched the API, engine, and raw SSE token counts and timings. Request SHA-256 hashes and summary statistics also matched. Re-run the audit with `python verify_results.py`. The live progress file counts the warm-up in `completed_runs`; summary statistics exclude it.

The saved memory snapshots are observations, not peaks. Windows HIP telemetry did not report GPU temperature or power, so neither is claimed. Q2_0 remains loaded at `http://127.0.0.1:8081/` after the benchmark.

## Limits

This is a long-prompt throughput test with synthetic code-review inputs, not an answer-quality or coding-task evaluation. The processor-group-aware binary is a local build rather than an official release executable. The interactive desktop and background applications were not isolated, and the gfx1030 run used plain hipBLAS without a tuned table. These measurements should not be presented as a general RX 6900 XT or official-release performance guarantee.
