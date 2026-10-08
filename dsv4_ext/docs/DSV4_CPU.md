# DeepSeek-V4 CPU MoE path

The optional CPU MoE path evaluates selected routed experts on the host while the
device path handles the remaining experts. It is opt-in at run time: the default
`--cpu-experts 0` keeps experts on the existing device-staging path.

## Work scheduling

The CPU path owns a persistent `CpuPool`; it does not create and destroy a thread
team for each layer. For each MoE invocation it flattens the selected experts'
rows into one range and divides that range into balanced static chunks. The first
phase computes gate and up rows and forms the weighted SwiGLU activations. Once
that phase completes, the second phase computes down-projection rows. This gives
the phases the barrier they require without scheduling individual experts as
separate tasks. Results are collected in fixed routing order, independent of
worker completion order.

In native mode, each input activation is quantized once for the corresponding
ggml activation format and reused by all selected experts that use that format.
The intermediate activation is quantized once per selected expert for the down
projection. The row-dot kernels and activation formats come from the pinned
ggml-cpu revision used by this extension. This follows the Qwen native-expert
precedent in Strata: use a pinned native CPU kernel implementation for expert
rows while keeping scheduling in the model path (see
[`native_expert.hpp`](../../include/strata/kernels/cpu/native_expert.hpp)). The
reference path uses the extension's float-activation row-dot implementation.

Native mode changes arithmetic: activations are quantized before ggml's native
row-dot kernels, so results need not be bit-identical to the reference path.
`--cpu-moe-kernel reference` remains available for the float-activation path.
The native path requires an x86 CPU and OS state supporting AVX2, FMA, and
F16C. `DSV4_NATIVE_CPU` and `DSV4_CPU_BMI2` are both off by default; BMI2 is
optional and can be slower on some Zen CPUs. `DSV4_CPU_AVX512` is a separate
opt-in for an AVX-512F/BW/VL/VNNI IQ1_M row-dot kernel. The pinned ggml kernels stay
at the AVX2 floor as a fallback. The override checks CPUID and XCR0 state `0xE6`
before dispatch; when the CPU or OS lacks AVX-512F, BW, VL, or VNNI, the native
AVX2 kernel remains available. It requires `DSV4_NATIVE_CPU=ON`. AVX-512 does
not change the quantized activation format. Its 256-bit VNNI path preserves the
pinned ggml AVX2 integer and floating-point accumulation order, and tests verify
bitwise row-dot parity for activations produced by ggml's Q8_K quantizer. An
injected `-128` activation byte is outside that quantizer's `[-127, 127]` range;
the optimized path retains ggml's signed-byte wrap behavior for this invalid
fixture. The build does not globally enable fast-math.

For IQ1_M dispatch validation, set `DSV4_CPU_IQ1M_KERNEL=avx2` or `avx512`
before starting the process. The default `auto` selects AVX-512 only for IQ1_M
rows of width 4,096 when the build and CPU/OS guard support it; other types and
widths use AVX2. This is the gate/up width used by the tested DeepSeek-V4 Flash
model. On the Ryzen AI MAX+ 395, two matched 16-thread HIP+CPU inference runs per
mode, with GPU settings held fixed, produced medians of 131.16 s for AVX2 and
128.07 s for VNNI. Repeat variation was about 5%, so this small sample does not
establish a robust end-to-end speedup.
A forced AVX-512 choice reports an error if the build or CPU/OS guard does not
support it. `DSV4_CPU_TRACE_DISPATCH=1` logs the selected kernel once when each
native row-width entry is first resolved, without adding per-dot counters.
Compare forced modes with the same prompt and model, and inspect logits as well
as generated text. Native mode can differ from the float reference because it
quantizes activations; the IQ1_M AVX2 and AVX-512 paths are tested for bitwise
parity on the same prepared Q8_K activation.

The native CPU test binary also supports `--bench-iq1m <width> [rows]`. It
cycles through distinct IQ1_M rows, checks selected output against an
independent dequantized-activation dot, and reports differences from pinned
ggml's AVX2 row-dot. The default 2,048 rows model one expert's resident gate or
up matrix; larger row counts can measure streaming weights beyond cache.
The MoE test binary supports `--bench 16 --pin --warmup 50 --iterations 400
--sets 8` for sustained pinned measurements with a rotating 267 MiB expert
weight set; omit the extra arguments for the short smoke benchmark.

## Runtime options

```text
--cpu-experts N             evaluate the last N routed experts on the CPU (0 = device staging)
--cpu-on-cache-miss          send VRAM-resident selected experts to GPU and cache misses to native CPU; requires --cpu-experts K
--cpu-threads N             persistent CPU MoE participants including the caller (0 = follow --threads)
--cpu-moe-kernel auto|native|reference
--no-cpu-pin                disable physical-core affinity for CPU expert workers
--no-cpu-host-pin           disable the scoped Windows CPU Set selection for the caller
```

`auto` chooses native kernels only when they were built, the current CPU/OS
passes the runtime ISA checks, and every routed expert weight format in the
model is supported; otherwise it uses the reference kernel. Selecting
`native` explicitly reports an error if any of those requirements are not met.
`--cpu-on-cache-miss` is a separate opt-in routing policy: it requires the GPU
backend, a working native CPU MoE backend, and `--cpu-experts` equal to the
model's routed expert count. Resident experts use the GPU; a nonresident expert
is evaluated on the CPU rather than copied through the device staging pool. A
free VRAM slot may still admit a selected expert once. `--expert-profile`
preloads the hottest measured experts before inference, while
`--expert-adapt-swaps` can promote later CPU misses between tokens. With this
policy enabled, the staged expert H2D counter should remain zero. The default
off path preserves the existing last-N CPU / remaining-device behavior.
The opt-in `DSV4_NATIVE_CPU=ON` build must configure its own pinned ggml targets
in the standalone `dsv4_ext` CMake build. It rejects pre-existing ggml targets
because their compiled ISA cannot be verified against the runtime AVX2 floor.
The CPU pool's participant count includes its calling thread and is separate
from the OpenMP threads selected with `--threads`. For example, 32 participants
means one caller plus 31 persistent workers; 64 means one caller plus 63
workers. With pinning enabled, each worker is hard-pinned to one logical
processor on a distinct physical core, with the first core reserved for the
caller. On a 3990X, the 64-participant layout is 31 workers in Windows processor
group 0 and 32 in group 1; the caller is separate from those worker counts.

On Windows, the runner holds the caller's reserved-core selection across one
token's forward pass with `SetThreadSelectedCpuSets`. Nested pool phases reuse
that selection, avoiding repeated OS migrations at every layer. Direct pool
calls also work without an outer scope. When the scope ends, the prior selected
CPU-set IDs are restored, including an originally empty selection. The caller is not
assigned a `SetThreadGroupAffinity` mask, so its existing Windows group
affinity—including the implicit all-group affinity on Windows 11—is left intact.
`--no-cpu-host-pin` disables only this scoped caller selection; `--no-cpu-pin`
disables worker pinning and also prevents the caller selection. Caller CPU-set
selection is Windows-only; Linux keeps its existing worker affinity behavior
and does not pin the caller.

Windows worker logical CPU IDs in logs are encoded as `group * 64 + processor`.
Use `--no-cpu-pin` to disable physical-core worker affinity when comparing
unpinned scheduling.

## HIP/CUDA matvec policy

`DSV4_MATVEC_THREADS` is the CMake-configured fallback block width for HIP or
CUDA matvecs. It defaults to 256 and accepts 32, 64, 128, 256, or 512. At run
time, `DSV4_MATVEC_POLICY` defaults to `fixed`, which uses that configured width
for every matvec. Set it to `shape` to use 32 threads for only these exact
DeepSeek-V4 Flash matrices:

```text
Q8_0: rows=32768, input width=1024
Q4_K: rows=129280, input width=4096
```

Every other matrix keeps the CMake fallback width. The selected shapes use 32
physical threads while each emulates eight of the original 256 virtual
accumulators. The register fold preserves the 256-wide accumulation and
reduction order; a same-binary real-model comparison produced bit-identical
logits, generated IDs, and routing traces. The policy value is checked once
during GPU initialization; any value other than `fixed` or `shape` fails
initialization. The selected mode and fallback width appear in the startup log.

On a Radeon 8060S (`gfx1151`), one matched 128-input/128-output run measured
124.77 seconds for `shape` and 130.10 seconds for `fixed`, with identical
32,966,400 logits and all 128 generated IDs. This is a single timing pair, not
a statistically conclusive speedup estimate. The measurements include GPU
kernel launch and output transfers, and do not establish results on other GPU
architectures. The serving launcher opts into `shape`; the engine default
remains `fixed`.

## Windows HIP build example: Threadripper 3990X

From the repository root in PowerShell, with the existing Strata TheRock
environment initialized and the Windows C++ build tools, CMake, and Ninja
available:

```powershell
$env:ROCM_VENV = "$PWD\.rocm-win"
$env:DSV4_HIP_ARCHS = "gfx1030"
$env:DSV4_NATIVE_CPU = "ON"
$env:DSV4_CPU_BMI2 = "OFF"
$env:BUILD_DIR = "$PWD\dsv4_ext\build-hip-3990x-native"
$env:BUILD_JOBS = "32"
cmd.exe /c dsv4_ext\build_windows_hip.bat
```

The batch file uses the pinned llama.cpp/ggml source when `DSV4_GGML_DIR` is
unset. If reusing an existing llama.cpp checkout, set `DSV4_GGML_DIR` to that
checkout before building. This example only configures and builds; it makes no
throughput claim. On a 3990X, verify the startup log's CPU MoE participant count
and worker CPU IDs before recording a run.

For a run that explicitly selects six CPU experts and the native kernel, append
options such as these to the normal HIP runner command:

```text
--cpu-experts 6 --cpu-threads 32 --cpu-moe-kernel native
```

Those values are configuration examples, not measured recommendations. Compare
with `--cpu-moe-kernel reference`, and report the exact model, prompt, thread
counts, cache state, and output token count for any performance result.
