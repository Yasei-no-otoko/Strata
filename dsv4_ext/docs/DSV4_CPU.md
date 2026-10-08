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
The native path requires an x86 CPU and OS state supporting AVX2, FMA, and F16C.
`DSV4_NATIVE_CPU` and `DSV4_CPU_BMI2` are both off by default; BMI2 is optional
and can be slower on some Zen CPUs. The build does not globally enable
fast-math.

## Runtime options

```text
--cpu-experts N             evaluate the last N routed experts on the CPU (0 = device staging)
--cpu-threads N             persistent CPU MoE participants including the caller (0 = follow --threads)
--cpu-moe-kernel auto|native|reference
--no-cpu-pin                disable physical-core affinity for CPU expert workers
--no-cpu-host-pin           disable the scoped Windows CPU Set selection for the caller
```

`auto` chooses native kernels only when they were built, the current CPU/OS
passes the runtime ISA checks, and every routed expert weight format in the
model is supported; otherwise it uses the reference kernel. Selecting
`native` explicitly reports an error if any of those requirements are not met.
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
$env:BUILD_JOBS = "4"
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
