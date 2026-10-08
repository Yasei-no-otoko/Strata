# DeepSeek-V4-Flash Windows HIP: archived diagnostic runs

These are historical experiments, including discarded implementations. For the final source, exact 128+128 comparison, verified affinity layout, and arithmetic limits, use [README.md](README.md). Statements below about a pending final comparison describe the stage at which these diagnostics were recorded.

These records are short diagnostics for a real pinned DeepSeek-V4-Flash UD-IQ1_M model on Windows HIP. They are not a final controlled exact-128-token optimization comparison. The separate initial exact-128 baseline and reproducible fixture remain in [the Windows HIP report](../2026-10-08-deepseek4-flash-windows-hip/README.md#exact-128-token-speed-measurement); a final controlled comparison belongs in a separate evidence section after validation.

## Machine and model

- Windows 11 Pro; AMD Ryzen Threadripper 3990X, 64 cores / 128 logical processors; 128 GiB physical RAM.
- AMD Radeon RX 6900 XT, gfx1030 / wave32, 16 GiB VRAM.
- AMD ROCm 10.2.0a20260930, AMD Clang 24, Visual Studio 2022 headers/linker, LLVM `libomp`.
- Pinned model: DeepSeek-V4-Flash UD-IQ1_M, three GGUF shards. The model manifest and checksums are recorded in the linked Windows HIP report.

## Shared diagnostic workload and settings

Each run used the same 32 prompt token IDs and requested 16 generated tokens, with context 512. This short prompt was selected for repeated kernel/runtime diagnostics, not as a substitute for the separate exact-128 fixture. The shared runtime settings were `--expert-ram profile` (no profile file, so the log confirms no experts copied to RAM), `--ram-cache-mib 0`, `--expert-cache 256`, and `--expert-vram-reserve-mib 4096`. The HIP logs identify the RX 6900 XT and `device: hip`.

The baseline build (`build-hip-baseline-2b67cdb`) was measured at 16 threads under its recorded environment, plus a passive-wait 16-thread run and a 1-thread run. The local optimization build (`build-hip-3990x`) was measured at 16 threads with `OMP_WAIT_POLICY=PASSIVE`, `KMP_BLOCKTIME=0`, and `--profile-timing`. Full commands, runtime metadata where available, memory samples, stdout, and stderr are preserved in each run directory below.

## Results

| Build / run | Prompt processing | Decode | Whole child process | Notes |
| --- | ---: | ---: | ---: | --- |
| Baseline, 16 threads | 32 tokens / 46.66 s (0.69 tok/s) | 16 / 14.45 s (1.11 tok/s) | 82.060 s | Default recorded baseline. |
| Baseline, passive 16 | 32 / 31.74 s (1.01 tok/s) | 16 / 11.53 s (1.39 tok/s) | 58.171 s | Same binary and workload; passive-wait comparison. |
| Baseline, 1 thread | 32 / 33.02 s (0.97 tok/s) | 16 / 14.82 s (1.08 tok/s) | 63.790 s | Thread-count diagnostic only. |
| Local optimization, wave AVX, 16 | 32 / 29.39 s (1.09 tok/s) | 16 / 11.44 s (1.40 tok/s) | 55.592 s | Generated token IDs/text differ from baseline on this short run; not quality/parity evidence. |
| Local optimization, grouped fused, 16 | 32 / 22.89 s (1.40 tok/s) | 16 / 8.15 s (1.96 tok/s) | 45.719 s | Output IDs match baseline for this short run. |

These are individual diagnostic runs, not medians, and the output lengths are too short for a robust throughput claim. The grouped-fused build is faster in this particular comparison, but root should confirm it under the exact same 128-token fixture and controlled run protocol before treating the change as a measured optimization. An isolated-kernel exploratory log reported regressions in 8 of 10 kernels for the wave-AVX experiment; this is a kernel-level diagnostic, not a substitute for end-to-end results.

## Expert-transfer observations

The baseline records show 458 hits and 11,668 misses (3.78%) over these 32 prompt plus 16 generated tokens, with 256 initial VRAM slots and no swaps. The optimization builds log about 78.8 GiB of staged expert H2D traffic over the same short workload and zero CPU expert evaluations. `CPU miss time` / `host submission/CPU fallback` includes host-side staging and submission work; the phase timings are nested and must not be summed. The zero-MiB LRU setting does not establish SSD rereads: mapped model files can be served from Windows' system file cache. Working-set and system available-memory samples in `summary.json` are retained to help interpret this distinction.

## Preserved raw records

- [Baseline 16-thread default](baseline-default16/)
- [Baseline passive 16-thread](baseline-passive16/)
- [Baseline 1-thread](baseline-default1/)
- [Local optimization wave-AVX 16-thread](optimization-wave-avx16/)
- [Local optimization grouped-fused 16-thread](optimization-grouped-fused16/)

Each run directory contains the original `stdout.txt`, `stderr.txt`, and `summary.json` copied from its build directory. The summary keeps the full process-memory sample trace. Original source records were left untouched.

## CPU MoE build and synthetic kernel sweep

The `build-hip-cpu-nobmi2` configuration enabled the native ggml CPU path with
BMI2 disabled, alongside the HIP backend. Its seven CTest tests passed, including
`cpu_pool`, `cpu_moe`, and `native_cpu`; the captured test log is
[`validation/ctest-7.log`](validation/ctest-7.log). The Windows processor-group
topology parser was corrected to validate variable-sized core records rather
than requiring the size of the largest union record. The CPU-pool test passed
with that parser in the build under test.

The pinned CPU MoE sweep below is a synthetic, cache-hot fixture
(`dim=4096`, `ff=2048`, six experts, IQ1_M gate/up and IQ2_XXS down,
33.38 MiB of weights). It is useful for validating the row-pool path and thread
scaling only; it is not model inference or an end-to-end speed result.

| Participants | Average native CpuMoe | Checksum |
| ---: | ---: | ---: |
| 1 | 25.89 ms | -44812201.7 |
| 8 | 3.40 ms | -44812201.7 |
| 16 | 1.83 ms | -44812201.7 |
| 32 | 1.89 ms | -44812201.7 |
| 64 | 1.30 ms | -44812201.7 |

Each case used one warmup and three timed iterations. The log also records the
sequential native-quantized comparison (about 28 ms) and the original float
reference (about 546 ms); these are different arithmetic paths and should not
be read as an end-to-end model comparison. The pinned sweep output is preserved
in [`validation/cpu-moe-pinned-sweep.log`](validation/cpu-moe-pinned-sweep.log).
The native CPU build reported ggml 0.24.0 at commit `3cf03257f`, with the
AVX2+FMA+F16C floor. BMI2 remained off for this run and is off by default; an
earlier BMI2-enabled microbenchmark used a different MSVC/unpinned setup, so its
results cannot isolate BMI2 as the cause of any timing difference.

The exact-128 final comparison remains separate; the completed short runs below
are not a substitute for that controlled fixture.

## Windows caller-affinity readback before caller pinning

The captured OS readback below is from the version before the run-scoped caller
CPU-set pin was added. It is baseline evidence only; it does not establish the
new behavior or imply that a post-change check has passed. The raw record is
[`validation/affinity-32-64-before-host.log`](validation/affinity-32-64-before-host.log).

| Requested participants | Caller readback before change | Worker layout and observed overlap |
| ---: | --- | --- |
| 32 | No selected CPU sets; group 0 mask `0xffffffffffffffff`; running at group 0 processor 42 | 31 pinned workers in group 0; worker `r21` also ran at 0:42 |
| 64 | No selected CPU sets; group 0 mask `0xffffffffffffffff`; running at group 0 processor 38 | 31 pinned workers in group 0 and 32 in group 1; worker `r19` also ran at 0:38 |

The individual worker masks in the captured log are single-bit masks, and their
group/processor readbacks are distinct physical-core primaries. The caller's
all-ones group-0 mask and empty CPU-set selection explain how it could be
scheduled on a worker's processor even though the worker masks themselves were
correct. A post-change run must verify the temporary caller CPU-set target and
restoration separately; no after-change pass is recorded here.

## Windows caller CPU-set readback after the change

The post-change log records the caller selected onto CPU Set 256, which maps to
logical CPU 0, inside both the 32- and 64-participant callbacks. The previous
selection was empty and was empty again after each run. The caller's group
affinity mask remained `0xffffffffffffffff`; this change selected a CPU Set for
the run without replacing that group affinity. Worker masks remained one bit
each: 31 workers in group 0 for the 32-participant case, and 31 in group 0 plus
32 in group 1 for the 64-participant case. CPU 0 no longer overlapped a worker
in these readbacks. The captured test output is
[`validation/affinity-32-64-with-host.log`](validation/affinity-32-64-with-host.log).
The matching seven-test run passed, including the pool, MoE, native CPU, and GPU
tests: [`validation/ctest-with-host.log`](validation/ctest-with-host.log).

## Tiny native-MoE arithmetic and serve-protocol checks

The tiny native activation oracle covered 26 by 40 logits. The native-mode
NumPy oracle matched the current native `CpuMoe` output with maximum absolute
difference `7.15255737e-7` and argmax agreement 26/26; the C++ output was also
byte-identical to the captured native tiny output. The existing float-reference
comparison reported maximum absolute difference `8.34e-7`, argmax agreement
26/26. Comparing the native-mode and float-mode oracle outputs gave maximum
absolute difference `0.867358208` and argmax agreement 24/26, as expected from
their different activation arithmetic. These tiny synthetic comparisons do not
establish real-model answer quality or quality equivalence. Raw records:
[`validation/native-oracle-validation.txt`](validation/native-oracle-validation.txt),
[`validation/tiny-reference.compare.txt`](validation/tiny-reference.compare.txt).

The API subprocess protocol check passed READY, a complete generation, partial
STOP without a newline, continued progress, cancellation, a subsequent
generation, and QUIT. Its short log is
[`validation/serve-protocol.log`](validation/serve-protocol.log).

## Completed short real-model CPU/HIP diagnostics

These are single 32-prompt-token / 16-generated-token diagnostics with context
512 on the pinned UD-IQ1_M model. All use OpenMP `--threads 16`, native ggml CPU
MoE kernels where applicable, `--ram-cache-mib 0`, and the recorded diagnostic
environment. They were run **before** the scoped caller CPU-set pin and before
the shared reference SwiGLU correction. Do not treat them as final optimized
results, repeated measurements, answer-quality evidence, or as a substitute for
the exact-128-token comparison. The cache-slot setting differs in two rows, as
shown.

| Diagnostic | CPU experts / CPU participants | Expert cache slots | Prompt, 32 tokens | Decode, 16 tokens | Process wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU six experts, 8 participants | 6 / 8 | 0 | 17.91 s (1.79 tok/s) | 7.64 s (2.09 tok/s) | 39.682 s |
| CPU six experts, 16 participants | 6 / 16 | 0 | 15.95 s (2.01 tok/s) | 6.19 s (2.58 tok/s) | 36.468 s |
| CPU six experts, 32 participants | 6 / 32 | 256 | 14.10 s (2.27 tok/s) | 6.05 s (2.65 tok/s) | 34.707 s |
| CPU six experts, 64 participants | 6 / 64 | 256 | 15.20 s (2.10 tok/s) | 10.77 s (1.49 tok/s) | 40.894 s |
| Hybrid, three CPU experts, 32 participants | 3 / 32 | 256 | 18.77 s (1.70 tok/s) | 7.36 s (2.17 tok/s) | 40.465 s |
| HIP-only async staging | 0 / 32 configured | 256 | 22.42 s (1.43 tok/s) | 8.02 s (2.00 tok/s) | 45.498 s |

The 32-participant CPU run logged 12,126 CPU expert evaluations and 0.000 GiB
staged expert H2D. The GPU-only row logged 458 hits and 11,668 misses. Run
stdout, stderr, and complete `summary.json` process-memory traces are preserved
under [`diagnostics/`](diagnostics/). Each case exited successfully, but the
differences in runtime settings and the pre-correction arithmetic limit direct
comparisons. No exact-128 result is reported here.

## Scoped caller CPU-set prototypes

These additional runs used the same short 32+16 workload to compare two
prototype placements for run-scoped caller CPU-set pinning. They are short
diagnostics, not the final exact-128 comparison or a quality measurement.

| Prototype | Participants | Prompt, 32 tokens | Decode, 16 tokens | Process wall |
| --- | ---: | ---: | ---: | ---: |
| Outer inference scope | 32 | 15.73 s (2.03 tok/s) | 6.72 s (2.38 tok/s) | 36.151 s |
| Outer inference scope | 64 | 16.03 s (2.00 tok/s) | 11.26 s (1.42 tok/s) | 41.131 s |
| Host scope | 32 | 24.82 s (1.29 tok/s) | 8.10 s (1.97 tok/s) | 48.331 s |
| Host scope | 64 | 17.08 s (1.87 tok/s) | 6.78 s (2.36 tok/s) | 38.523 s |

The outer-scope 64-participant run decoded at 1.42 tok/s versus 2.38 tok/s
with 32 participants in these individual runs. This does not establish a general
thread-count result. The records are preserved in
[`diagnostics/outer-pin32/`](diagnostics/outer-pin32/),
[`diagnostics/outer-pin64/`](diagnostics/outer-pin64/),
[`diagnostics/host-pin32/`](diagnostics/host-pin32/), and
[`diagnostics/host-pin64/`](diagnostics/host-pin64/). Each directory contains
stdout, stderr, and the completed summary only.

The final affinity readback verified both participant counts with CPU Set 256
(logical CPU 0) selected for the caller during the callback, then restored the
previous empty CPU-set selection. The all-group group-affinity mask remained
unchanged. Worker readbacks retained distinct one-bit masks: 31 workers in
group 0 for 32 participants, and 31 in group 0 plus 32 in group 1 for 64.
See [`validation/affinity-final.log`](validation/affinity-final.log); the
matching seven-test CTest run passed at
[`validation/ctest-outer-scope.log`](validation/ctest-outer-scope.log).

The post-fix precision comparison checked 15 by 129,280 logits. GPU HIP output
was bit-exact against the reference (maximum absolute difference 0; argmax
agreement 15/15). Native CPU Q8 output differed from the float reference
(maximum absolute difference 5.045, RMS 0.494, argmax agreement 14/15).
These are logit-level arithmetic comparisons, not answer-quality evidence or a
claim of quality equivalence. Raw data:
[`validation/quality-final-comparison.json`](validation/quality-final-comparison.json).
