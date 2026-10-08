# DeepSeek-V4-Flash Windows HIP port: RX 6900 XT / 128 GiB

This work ports upstream [PR #1518](https://github.com/Niko1221/Strata/pull/1518),
head `1e28cd54dcf7866e79afb3e5b7b2024f73d0d654`, in a separate worktree.
The original Swift installation is preserved. Date: 2026-10-08, JST.

## Machine and build

- Windows 11 Pro, AMD Ryzen Threadripper 3990X (64 cores, 128 logical processors).
- 128 GiB physical RAM (127.876 GiB visible to Windows).
- AMD Radeon RX 6900 XT, gfx1030 / wave32, driver `32.0.21045.5002`.
- ROCm `10.2.0a20260930`, AMD Clang 24, Visual Studio 2022 headers/linker,
  local VS LLVM OpenMP runtime; target `gfx1030`.
- Standalone extension build: `DSV4_WITH_HIP=ON`, `DSV4_WITH_CUDA=OFF`, Release.
- Original unmodified PR does not build with MSVC: POSIX headers, `__int128`,
  and `M_PI` are among the failures. It has no HIP backend.

## Port regressions

| Check | Result |
| --- | --- |
| Windows MSVC CPU build, CTest | 3/3 passed |
| Linux WSL GNU 13.3 CPU build, CTest | 3/3 passed |
| Native Windows HIP build, CTest | 4/4 passed |
| Actual HIP matvecs | F32, BF16, Q8_0, Q4_K, Q5_K, Q6_K, IQ3_XXS, IQ2_XXS, IQ1_M, MXFP4 match CPU within test tolerance |
| Actual HIP MoE | resident and host-staged experts match the CPU reference |
| Synthetic complete forward on HIP | 26/26 reference argmax agreement; max absolute logit difference 1.19e-6, relative 6.83e-7 |
| Synthetic HIP with zero resident expert slots | same 26/26 agreement, proving the staging path through the model |
| No visible HIP device | engine exits 1 with `cannot initialize hip`; no CPU fallback |
| Windows/Linux serving pipe | complete request, partial STOP without newline, progress, cancellation, next request, QUIT |
| Large mapped-file offset | sparse sentinel at 4 GiB + 12345 read correctly on Windows and Linux |
| Profile replacement | existing destination atomically replaced after successful writes and sync |

The synthetic fixture is the PR's `tools/tiny_model.py`: four layers, eight experts,
40 vocabulary entries, 26 input positions. It is not the real model. Its generated
`t32` text and displayed one-token decode rate are not useful model-quality or
performance evidence. GPU tests refuse the CPU-emulated device backend.

The HIP log identifies `AMD Radeon RX 6900 XT (arch gfx1030, wave32)` and reports
`device: hip`. CPU/WSL evidence is recorded separately from real GPU execution.

## Reproduce

See [the extension's Windows instructions](../../../dsv4_ext/README.md#windows-hip-experimental)
for build/runtime prerequisites. From the repository root in a command prompt:

```bat
set ROCM_VENV=C:\path\to\Strata\.rocm-win
set DSV4_HIP_ARCHS=gfx1030
dsv4_ext\build_windows_hip.bat
dsv4_ext\run_windows_hip.bat ctest --test-dir dsv4_ext/build-hip-win --output-on-failure
python dsv4_ext/tools/tiny_model.py dsv4_ext/build-hip-win/tiny
dsv4_ext\run_windows_hip.bat python dsv4_ext/tests/test_serve_protocol.py dsv4_ext/build-hip-win/dsv4_run.exe dsv4_ext/build-hip-win/tiny/tiny.gguf
```

For full-forward parity, pass `tiny/tokens.txt` as `--prompt-ids`, use `--ctx 64
--n-predict 1 --no-qat-sim --threads 4 --dump-logits <output.bin>`, and compare with
`tools/compare_logits.py <tiny/ref_logits.bin> <output.bin> 40`. Add `--expert-cache 0
--ram-cache-mib 0` for the staged path. The protocol test deliberately uses CPU math
to isolate pipe behavior; the GPU and forward tests provide device evidence.

## Real model: native Windows HIP

Unsloth `DeepSeek-V4-Flash-GGUF`, revision
`e3aa0d6a5fa4f820d9e132ac1fd1d01e1b2b49e0`, UD-IQ1_M: three shards,
86,901,313,152 bytes (80.93 GiB). All three sizes and SHA-256 hashes match the
pinned model repository; see [the verified manifest](model-manifest.json).
Metadata identifies architecture `deepseek4`, 43 layers, 256 experts/top-6,
tokenizer preprocessor `joyai-llm`, and expert gating function ID 4.

The complete model ran on the RX 6900 XT with context 512, 16 CPU threads,
256 resident expert slots, `--expert-ram profile` without a profile,
`--ram-cache-mib 0`, and `--expert-vram-reserve-mib 4096`. No layer limit,
CPU-only option or QAT override was used. MTP was not tested.

| Short arithmetic request | Measured result |
| --- | --- |
| Prompt | `What is 2 + 2? Answer briefly.`; 15 template/tokenizer tokens |
| Generated text | `2 + 2 = 4`, followed by the GGUF EOS token |
| Model load | 6.1 s, `loaded ... (hip)` |
| Prompt processing | 14.89 s, 1.01 token/s |
| Generation | 8 tokens including EOS, 5.59 s, 1.43 token/s as reported by the runner |
| Whole process | 32.495 s, exit 0, including selfcheck/logit dump and cleanup |
| Peak process working set | 20.47 GiB |
| Peak sampled private bytes | 0.322 GiB |
| Minimum sampled system available RAM | 66.66 GiB of 127.876 GiB visible |
| Device allocation at load | 7.86 GiB of 15.98 GiB; driver reported 7.76 GiB free |
| Tensor selfcheck | all ten sampled quantization types finite |
| Forward logits | all finite: 22 positions x 129280 vocabulary entries |

Raw [stdout](real-short/stdout.txt), [backend/load log](real-short/stderr.txt),
and [exact command plus 0.2 s memory samples](real-short/summary.json) are retained.
The one-shot runner includes the first sampled token and EOS in its decode count;
this short rate is not a sustained benchmark. Checksums had just read the shards,
so the filesystem cache could be warm. The mapped files total 80.93 GiB: the
20.47 GiB working set means only some mapped pages were resident in this request,
not that the whole model occupies that amount. Private bytes exclude file-backed
pages and device memory. The VRAM number is an allocation log at load, not a
driver-wide peak measurement. The planner's printed `RAM expert copy` is the
inventory size; the actual loader confirms zero copied experts and zero LRU slots.

The JoyAI adapter fixes number splitting and supplies the template's declared BOS.
Four adapter regressions pass; six actual-vocabulary cases match the independent
Rust BPE implementation, and the rendered BOS is ID 0. See
[tokenizer output](tokenizer-oracle.txt). The regex itself follows pinned llama.cpp
source; this comparison does not independently test the regex implementation.

These measurements establish short-context operation on this 128 GiB
machine; they do not establish 128k/1M context, broad model quality, tool calling,
vision, other AMD cards, or CUDA runtime parity.

## API checks

The dedicated server ran on `127.0.0.1:8095`, with the same memory/thread options
and context 512. The initial lazy health check correctly reports context 512;
the engine then reports `device=hip`, all 43 layers, and `READY 512`.

| Request | Visible answer | Result | Wall time |
| --- | --- | --- | --- |
| Chat Completions, default reasoning settings | `4` | HTTP 200 | 22.988 s including lazy model load |
| Chat Completions, Japanese SSE | `東京都` | HTTP 200, UTF-8 chunks, `[DONE]` | 14.148 s; first content 12.679 s |
| Responses, a short English sentence | `The Moon is Earth's natural satellite, orbiting our planet and reflecting sunlight to create its phases.` | HTTP 200, visible `message/output_text` | 23.567 s |

The wrapper aligns the default no-thinking template prefix with the output parser,
so ordinary answers arrive in `content`, not only in `reasoning_content`.
The API engine's peak working set after the three requests was 35.82 GiB, with
0.322 GiB private bytes. Its status reported 76.4 GiB total system RAM used at
that point; this is a snapshot, not a sampled system peak.

See [requests, full responses and assertions](api-checks.json),
[engine log](api-engine.txt), [memory counters](process-memory-after-api.json),
[status](status-after-api.json), and [reproduction script](run_api_checks.py).
The script never starts a server: run it against the above dedicated local server.
Non-streaming first-content latency is the complete HTTP response latency, not TTFT.

## Exact 128-token speed measurement

The requested 4K run was cancelled before inference began. The replacement run
uses exactly **128 input tokens including BOS and the chat template**, and requests
128 generated tokens. Context capacity remains 512; it is not the prompt length.
The saved [input manifest](short-128-input/manifest.json), messages, rendered prompt
and token IDs are byte-verified. Re-encoding the saved rendered prompt produces
the same 128 IDs. [make_benchmark_prompt.py](make_benchmark_prompt.py) reproduces it.

| Measurement | Result |
| --- | --- |
| Input processing | 128 tokens / 138.43 s = 0.92 token/s |
| Generation | 128 tokens / 120.96 s = 1.06 token/s |
| Load | 6.2 s, native HIP on RX 6900 XT |
| Complete child process | 282.768 s, exit 0, including setup and cleanup |
| Peak process working set | 53.28 GiB |
| Peak sampled private bytes | 0.322 GiB |
| Minimum sampled system available RAM | 34.03 GiB |
| VRAM allocation at load | 7.86 GiB |
| Resident expert hit rate | 1,233 / 65,790 = 1.87% |

The output summarizes the fictional notes and reaches the requested generation
limit partway through its final section. This is a single run with no cache flush,
not a median or a cold-storage benchmark. The sampling and memory options match
the short run; this speed run does **not** enable `--selfcheck` or `--dump-logits`.
The earlier logit-dump run computes vocabulary logits at every prompt position,
so its prompt rate is not a directly controlled speed comparison.

See [stdout](real-128/stdout.txt), [HIP log](real-128/stderr.txt),
[exact command and memory samples](real-128/summary.json), and
[engine build identity](build.json). As in the earlier run, the runner's decode
counter includes the first token sampled from the final prompt logits, so 128
generated tokens correspond to 127 additional forwards. All 43 layers run.

## Performance assessment against upstream CUDA

The live [PR body](https://github.com/Niko1221/Strata/pull/1518), re-read at the
same head SHA, says UD-IQ1_M was tested with CUDA 12.8 on Ubuntu 22.04.5. It does
not give GPU model, prompt/output length, run flags or token/s. The CMake/build
comments mention an RTX 4090 / sm_89 target, but contain no runtime measurements.
PR reviews, conversation comments and inline review comments were empty. The
[saved PR metadata](upstream-pr.json) records the body and head used here.

The [PR-head README](https://github.com/Niko1221/Strata/blob/1e28cd54dcf7866e79afb3e5b7b2024f73d0d654/dsv4_ext/README.md)
mentions approximately 0.2 seconds/token in an OpenMP explanation, but provides
no CUDA run log or conditions for that number. Its verification table still marks
real-model inference and CUDA kernel execution as not run. The body and table are
inconsistent; neither establishes a reproducible practical CUDA throughput baseline.

Source inspection identifies common CUDA/HIP costs, not a measured HIP regression:

1. `model.cpp::mv` copies activations H2D, launches a matvec, then performs a
   blocking D2H for each projection. Activations, attention and transformer mixing
   mostly remain on the host.
2. `gpu_cuda.cu::experts_hit` copies mmap-backed experts to pinned host memory,
   queues all selected expert H2D transfers, then launches the kernels. All use
   the implicit default stream. This does not overlap the next expert's transfer
   with the previous expert's GPU computation as its comment claims.
3. The 1.87% hit rate measures VRAM residency. Supported misses use the HIP staging
   pool; it does **not** mean only 1.87% of expert computation runs on the GPU.
4. Sparse attention uses host OpenMP loops with double intermediates. Prefill
   calls the complete forward once per token, with no prompt-batched matrix multiply.
5. The common matvec kernel uses one 256-thread block per output row, repeated
   bit decoding and a shared-memory tree reduction. No architecture-specific
   wave reduction or matrix-library path is used.

In this run, `cpu_miss_s=110.39` measures a region containing staging, H2D calls,
kernel submission and any CPU fallback; it is not pure CPU math. `gpu_hit_s=41.95`
measures the remaining D2H completion wait; it is not total GPU kernel time. Those
two MoE regions total 152.34 s (58.7% of the 259.39 s prompt-plus-decode interval).
The remaining time also includes projections, host attention/mixing, sampling and
loop overhead. These coarse counters cannot isolate HIP driver overhead from
host work or transfer bandwidth. The reported H2D byte counter does not include
all staged expert transfers and must not be used as total PCIe traffic.

The HIP port retains the original kernel math, block size and launch order. Its
GPU-source changes are runtime mapping, backend/device reporting, error checks
and HIP-compatible fixed-size bit copies. Thus shared implementation costs are
credible optimization candidates; a backend-specific slowdown remains unmeasured.

No Ryzen 9950/AVX-512 or gfx1151 hardware was measured here. The user's condition
for extreme target optimization (verified practical CUDA speed) is not established,
so no target performance claim or unmeasured architecture optimization is made.
The next useful work would first separate host attention, staging, H2D, kernels
and D2H with timers/events, then keep activations on GPU, overlap staging with
separate streams/events, batch prefill, and tune CPU SIMD/GPU quant kernels on the
actual target. AVX-512 flags alone do not remove these common dataflow costs.

## Original service restored

The dedicated DeepSeek server and engine were stopped. The original Swift service
on `127.0.0.1:8081` was reloaded; health reports the original model, context 131072,
`loaded=true` and `images=true`. A real text request returned `OK`. The configuration
SHA-256 is unchanged and the original checkout remains clean. See
[restoration evidence](swift-restored.json). Image availability is a health flag;
no new image-inference test was performed in this task.
