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

## Verified before downloading the complete real model

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

## Real model status

Unsloth `DeepSeek-V4-Flash-GGUF`, revision
`e3aa0d6a5fa4f820d9e132ac1fd1d01e1b2b49e0`, UD-IQ1_M: three shards,
86,901,313,152 bytes (80.93 GiB). At this initial validation checkpoint the large
shards are still downloading. Full-model inference, peak RAM and API behavior are
not yet verified. The first shard's metadata identifies architecture `deepseek4`,
tokenizer preprocessor `joyai-llm`, and expert gating function ID 4.

The planned short-context test uses mmap weights, no profile RAM copy, no LRU RAM
copy, and reserved VRAM for the desktop. The measured process working set and system
available RAM will be recorded separately; model file size is not peak RAM use.
