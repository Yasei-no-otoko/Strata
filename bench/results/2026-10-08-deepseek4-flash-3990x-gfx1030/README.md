# DeepSeek-V4-Flash: 3990X CPU MoE and Windows HIP

On a Threadripper 3990X with an RX 6900 XT, optimized HIP generated 128 tokens
at **2.04 tok/s**, versus **1.40 tok/s** for the original Windows port.
Evaluating the six routed experts with the new AVX2 CPU pool increased this to
**2.27 tok/s**. These are individual serial measurements on one 128-token
prompt, not repeated-run medians or a general quality benchmark. Native CPU
MoE quantizes activations and changes generated output; it remains opt-in.

## Exact 128-input / 128-output comparison

| Configuration | Prompt, 128 tokens | Generation, 128 tokens | Whole process | Peak process working set | Minimum available system RAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original HIP port, `2b67cdb` | 112.45 s / 1.14 tok/s | 91.52 s / 1.40 tok/s | 223.605 s | 53.28 GiB | 32.10 GiB |
| Optimized HIP, `e5a62c9` | 70.87 s / 1.81 tok/s | 62.60 s / 2.04 tok/s | 153.239 s | 53.28 GiB | 32.74 GiB |
| Optimized HIP + native CPU MoE, 32 participants | 54.27 s / 2.36 tok/s | 56.41 s / 2.27 tok/s | 130.166 s | 53.13 GiB | 32.09 GiB |

Calculated from durations, HIP improves generation throughput 1.46x and CPU
MoE improves it 1.62x over the original port. CPU MoE is 1.11x optimized
HIP-only generation speed. Prompt processing improves 1.59x and 2.07x
respectively. The CPU speed comparison is not an equivalent-output claim.

All three processes exited successfully. The fixture contains exactly 128
input IDs including BOS and template tokens; context capacity is separately
512. Each actually generated 128 IDs. The first output is sampled from the
last prompt logits, so this means 128 prompt forwards plus 127 further
forwards (255 total). No logit dumping was enabled during timing. Whole-process
duration includes loading and exit, unlike the prompt/decode timers.

The fixture is
[`prompt-ids.txt`](../2026-10-08-deepseek4-flash-windows-hip/short-128-input/prompt-ids.txt),
SHA-256 `c773455ed7b77c46bead9e2624baa0c2ab294d0a3e2dd13f34fc18ca0c6aa8bd`.
Full commands, executable hashes, environment, memory samples and output are
in [`final128-baseline/`](final128-baseline/),
[`final128-hip/`](final128-hip/) and [`final128-native32/`](final128-native32/).
[`verify_results.py`](verify_results.py) checks exact token counts, real HIP
device evidence, exit status and native expert counters, and regenerates
[`results.json`](results.json).

## Machine, builds and conditions

- Windows 11 Pro; AMD Ryzen Threadripper 3990X, 64 cores / 128 logical CPUs,
  127.88 GiB physical RAM.
- AMD Radeon RX 6900 XT, gfx1030, wave32, 16 GiB VRAM; driver
  `32.0.21045.5002`; PCIe 4 x16 reported by Windows.
- TheRock ROCm `10.2.0a20260930`, AMD Clang 24 and VS 2022
  MSVC 14.44.35207 headers/linker; LLVM OpenMP runtime.
- DeepSeek-V4-Flash UD-IQ1_M, `unsloth/DeepSeek-V4-Flash-GGUF` revision
  `e3aa0d6a5fa4f820d9e132ac1fd1d01e1b2b49e0`. Three shards total
  86,901,313,152 bytes (80.93 GiB), including 74.42 GiB of expert weights.
  Model hashes are in the [initial port report](../2026-10-08-deepseek4-flash-windows-hip/README.md).
- Baseline executable SHA-256:
  `9688fad315fd89b3ed73c9c599a0a9d9ec67117b0304a50690d71e8f9a4f0b35`.
  Optimized source `e5a62c9783f2d2082c87a788ccb8dc453798cbe8`, executable
  SHA-256 `28faf81f2e5b4f90f8a97c34ac994385540bae576223029f3954a1d7af183eab`.
  The subsequent MSVC portability fix `9bb3584` replaces conditional scope
  construction with separate returns; it changes no numerical operation or
  scope lifetime. Its rebuilt HIP executable SHA-256 is
  `e9ad6d320fec27c157fc733f4ae663bc45313c7fb239da4cce1485b937b34b9a`.
  This rebuilt binary passed the final integration checks below; the timing
  table retains the original measured executable identity.
- Optimized build: `DSV4_WITH_HIP=ON`, `DSV4_AVX2=ON`,
  `DSV4_NATIVE_CPU=ON`, `DSV4_CPU_BMI2=OFF`, HIP target `gfx1030`.
  Pinned ggml revision `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
- Common runtime: `--ctx 512 --threads 16 --expert-ram profile
  --ram-cache-mib 0 --expert-vram-reserve-mib 4096`. No expert profile was
  supplied; logs confirm no explicit expert copy into RAM. HIP cases use
  256 expert cache slots; the CPU-six-expert case uses zero slots.
- `OMP_NUM_THREADS=16`, `OMP_WAIT_POLICY=PASSIVE`, `KMP_BLOCKTIME=0`,
  `DSV4_HIP_ASYNC_STAGE=0`; no OpenMP affinity variables set. CPU MoE has
  its own persistent pinned pool, separate from these 16 OpenMP threads.
- The original executable has no `--profile-timing` option. Both optimized
  runs enable nested phase timers and include their overhead. Thus the
  baseline comparison is not instrumentation-identical and does not isolate
  a single kernel change.
- Swift was unloaded during measurement. Cases ran serially in
  baseline/HIP/native order after prior diagnostics. No Windows page-cache
  flush was performed. These are warm-cache runs; zero explicit RAM cache
  does not mean every mapped page came from SSD.

The initial report's 1.06 tok/s baseline used a different OpenMP wait
environment. The controlled 1.40 tok/s baseline above is the comparison for
this optimization. VRAM logs describe load-time allocation (7.86 GiB HIP,
6.13 GiB CPU MoE), not peak device memory during inference.

## Processor Groups: 32 and 64 participants

Worker Affinity Masks **are applied in both cases**. The caller was not pinned
and could share a worker's core. The new scope follows Qwen's Windows CPU Set
approach: temporarily select the reserved caller core, then restore the exact
previous CPU-set selection. It preserves Windows 11's implicit all-group
affinity instead of replacing the caller's hard group mask.

| CPU participants | Caller | Persistent workers | Actual OS readback |
| ---: | --- | --- | --- |
| 32 | CPU Set 256, group 0 processor 0 | 31 in group 0, processors 2, 4, ..., 62 | One-bit masks and matching actual processors |
| 64 | CPU Set 256, group 0 processor 0 | 31 in group 0 plus 32 in group 1, processors 0, 2, ..., 62 | Both groups execute; one-bit masks match actual processors |

Counts include the caller: 32 means 1+31; 64 means 1+63. CPU Set IDs are
opaque OS values; 256 is this machine's observed ID, not a hardcoded constant.
The test reads `GetThreadGroupAffinity`, `GetCurrentProcessorNumberEx` and
selected CPU sets inside work callbacks. It verifies nested scopes and
restoration of the original empty selection. See
[`validation/affinity-final.log`](validation/affinity-final.log).
Before the fix, caller CPUs 42/38 overlapped workers; see
[`validation/affinity-32-64-before-host.log`](validation/affinity-32-64-before-host.log).

The Windows topology parser also needed correction: physical-core records
are variable-sized and must not be rejected because they are smaller than
the largest relationship union. The corrected pool selects physical-core
primaries rather than neighboring SMT threads.

The caller scope spans a whole token forward. Repeating CPU-set changes in
each layer's two pool phases was slower and was discarded. On the final
implementation's short 32-input/16-output diagnostic, 32 participants achieved
2.38 tok/s and 64 achieved 1.42 tok/s, so the final 128 run uses 32.
Group 1 workers were directly observed running. Cache, memory traffic and
synchronization across more workers are plausible limits; these runs do not
isolate their individual costs. See [`diagnostics/outer-pin32/`](diagnostics/outer-pin32/)
and [`diagnostics/outer-pin64/`](diagnostics/outer-pin64/).

## Implementation and remaining costs

The CPU path reuses Qwen's approach of native quantized expert dots with a
persistent row pool. Selected experts' gate/up rows form one balanced phase,
followed by a down-projection phase and fixed-order sum. Activation
quantization is shared per input format and per expert intermediate. It avoids
restarting thread teams and nested ggml/OpenMP teams. Native runtime checks
require AVX2, FMA, F16C and OS vector-state support; the portable reference
fallback remains available. BMI2 is optional and defaults off.

HIP changes group eight attention projection transfers into one, retain the
shared-expert intermediate on the GPU, preserve the original wave32 reduction
order, and interleave host preparation with queued work. Host sparse attention
and hyperconnection operations have an optional AVX2 implementation. Async
expert-copy staging was tested but is disabled for this comparison and by
default. No fast-math was added.

HIP-only staged **436.364 GiB** of routed-expert H2D traffic, with a 1.87%
expert-cache hit rate. CPU MoE performs **65,790 native evaluations**
(`255 * 43 layers * 6 experts`) and stages **0 GiB** of routed-expert weights.
Attention, the shared expert and other dense work still use HIP. Native CPU
records 55.639 s in MoE and 47.213 s in attention, including 44.171 s in
dense matvec round trips across multiple phases. Timers are nested and must
not be summed. Removing expert transfers leaves other host/device sync costs.

[Upstream PR #1518](https://github.com/Niko1221/Strata/pull/1518) described an
Ubuntu CUDA 12.8 test environment but supplied no GPU model or throughput
measurement in the inspected body. This cannot establish CUDA practicality
or that HIP alone causes slowness. Local profiles establish substantial costs
in transfer/synchronization and MoE execution. There is no CUDA,
Ryzen 9950/AVX512 or gfx1151 speed claim from these results.

## Numerical validation and limits

Optimized HIP generated the same 128 IDs as the baseline. Separately, all
15 x 129,280 checked real-model logits were bit-identical after preserving
reference shared-expert activation arithmetic. This supports the tested
cases, not an exhaustive equivalence proof.

Native CPU MoE quantizes activations and generated different IDs. Against
float-activation real-model logits, the separate 15-position check had
maximum absolute difference 5.045, RMS 0.494 and argmax agreement 14/15.
No accuracy, perplexity or answer-quality equivalence is established. Native
CPU use is explicitly opt-in; `--cpu-moe-kernel reference` retains float
activation dots. See [`validation/quality-final-comparison.json`](validation/quality-final-comparison.json).

The independent tiny NumPy oracle checks 26 x 40 logits. Native-activation
oracle versus native C++: maximum absolute error 7.15e-7, argmax 26/26.
Float oracle versus reference C++: maximum absolute error 8.34e-7, argmax
26/26. Comparing native directly to the float oracle differs by design.
Final build checks and protocol logs are under [`validation/`](validation/).

The HIP/native build passed [seven CTests](validation/ctest-outer-scope.log).
After the MSVC scope-construction fix, the portable build passed
[five CTests](validation/portable-final-ctest.log), and the rebuilt HIP runner
passed [async GPU, both tiny oracles and serve-protocol checks](validation/final-integration-validation.txt).
The protocol check covers READY, complete generation, partial STOP without a
newline, progress, cancellation, a subsequent request and QUIT. Native and
reference tiny outputs remained byte-identical to their pre-portability-fix
artifacts. CUDA hardware and AVX512-specific kernels were not tested.

## Original service restoration

After testing, the original `huihui-swift-1.5-abliterated-iq3_xxs` service was
restarted on `127.0.0.1:8081` using its existing configuration. Engine 0.1.40
loaded and a real chat request returned `OK`. The configuration SHA-256
remained `3213f66bd950da1376ceb56af91c37be95f402b1f228d124c0da9c6ed22292c8`.
The restored engine log confirms the RX 6900 XT / gfx1030 and HIP runtime;
see [restoration receipt](validation/swift-restored.json) and
[runtime excerpt](validation/swift-restored-runtime.txt).

## Reproduction

Build instructions and CPU options are in
[`DSV4_CPU.md`](../../../dsv4_ext/docs/DSV4_CPU.md). Also set `DSV4_AVX2=ON`
for these host-op optimizations. The development launcher uses installed local
ROCm and VS OpenMP; its local `debug_nonredist` runtime is not a
redistributable release package.

From the repository root, use the preserved full argument arrays in each
`summary.json`, including the exact prompt IDs. CPU-specific options:

```text
--cpu-experts 6 --cpu-threads 32 --cpu-moe-kernel native --expert-cache 0 --profile-timing
```

For HIP-only use `--cpu-experts 0 --expert-cache 256`. Run one process at a
time with the recorded environment. Verify existing results without a model:

```powershell
python bench/results/2026-10-08-deepseek4-flash-3990x-gfx1030/verify_results.py
```

Historical short sweeps, microbenchmarks and discarded variants are preserved
in [DIAGNOSTICS.md](DIAGNOSTICS.md). They do not replace the exact-token results.
