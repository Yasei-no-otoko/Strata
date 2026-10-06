# Strata v0.1.40 and CPU vision on Windows RX 6900 XT

On 2026-10-06 (Asia/Tokyo), this Windows installation was updated from Strata
0.1.39 to 0.1.40, restarted, and extended with a locally built CPU image
encoder. Huihui Swift IQ3_XXS remains the installed model, at
`http://127.0.0.1:8081`, with context 131072. Text and image requests passed on
the physical RX 6900 XT. No branch or tag was pushed.

## Release and preserved installation

The official [v0.1.40 release](https://github.com/Niko1221/Strata/releases/tag/v0.1.40)
provided `strata-windows-x64-hip.zip` (627,481,056 bytes).
Its complete local SHA-256 matched GitHub's asset digest:
`b2fa4a660409fa0f842d442a6991f359a7f0aebefec6f854c852a8b639621d7a`.
The installed `strata.exe` matches the executable inside the archive:
`e7605caf89b939e262ff9c0ca763aa99493277c9fa61d3cc9ee630f902b9ff96`.

The fetched official tag resolved to source commit
`1cbcacbcae2953f3be9edc46369f0c875bc6ab8b`. Git reported unrelated histories
when merging it into the previous local installation branch. A new branch
therefore starts at the official tag, with the two prior installation-record
commits cherry-picked. The earlier branch remains intact. No application,
engine, or installer source was modified locally.

The existing server was idle when stopped. Config, launcher, engine metadata,
and model metadata were backed up under `logs/update-v0.1.40-20261006/`.
The update used the new release's setup:

```text
.venv\Scripts\python.exe setup.py --update --yes --prebuilt logs\update-v0.1.40-20261006\release
```

Setup retained engine 0.1.39 under `engine/.previous/`. Both existing Swift
GGUFs have the same paths, byte sizes, and modification timestamps as before
the update. They were not downloaded or repacked. This is a preservation
check, not a fresh full checksum of the 76 GB model.

The model name remains `huihui-swift-1.5-abliterated-iq3_xxs`. Context 131072,
KV `int8`, resident KV 32768, speculative decoding 4, port 8081, and loopback
binding were preserved. The config changes consist only of appending the
engine's `--vision` flag and adding its `vision` block.

## Image encoder

The official Windows HIP archive has no image encoder, and `hip_vision()`
still turns ordinary Windows `--vision cpu` setup requests off. The existing
`setup.build_vision_cpu()` helper can build that target with MSVC, however.
It was invoked directly, without changing the installer or recompiling the
HIP inference engine. Visual Studio 2022, CMake, and Ninja were already
installed on this PC.

The helper built `tools/vision` with `STRATA_VISION_CUDA=OFF` against a fresh
copy of release-pinned llama.cpp commit
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. It installed
`engine/strata-vision.exe` and recorded `vision: cpu` in `engine/BUILD.json`.
The encoder SHA-256 is
`4bf57f1ef3868016321757f90370de555c28c94ab18bfac3e38f8d3dced7fdea`.

The projection model is
`mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf` (907,543,520 bytes), from the same
[Huihui revision as the text model](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF/tree/1e1bd8216ba46a590d8168cca08f68de9536e4ac).
Its full SHA-256 matched the pinned LFS metadata:
`cd1140f4abba943ce8d9fa92fed8407e389954d26907efde7b72153c75bd8a60`.

The vision config uses `gpu: false`, 64 CPU threads, and setup's CPU default
cap of 300 image tokens. Its vocabulary/model path is Swift's first GGUF
shard. Image encoding runs on the CPU; language-model inference continues
on the HIP engine. The manual CPU encoder addition was validated on this
machine; it is not a claim that the stock Windows installer supports it.

## Runtime checks

Hardware was rechecked: Threadripper 3990X (64 cores, 128 logical processors),
137,306,128,384 bytes of system RAM, RX 6900 XT (gfx1030, 16 GiB), and AMD
driver `32.0.21045.5002`.

| Check | Observed result |
| --- | --- |
| HIP device probe and engine startup | RX 6900 XT; `session is up (engine 0.1.40)` |
| `GET /health` | `loaded: true`, `images: true`, Swift model, context 131072 |
| `GET /props` | `build_info: Strata 0.1.40` |
| `GET /v1/models` | Existing Swift model identifier |
| `GET /` | HTTP 200, Strata HTML |
| English chat | `Strata update verified.`, finish reason `stop` |
| Japanese chat | A Japanese greeting, finish reason `stop` |
| Text Responses API | `Ready.`, status `completed` |
| Image chat and image Responses API | Both identified the red circle, blue square, and text `STRATA140` |

The [640 x 384 test image](vision-smoke-test.png) was generated locally.
The prompt asked for the shapes, their order, and the printed text, without
naming the expected content. Both image routes answered:

> The two shapes are a red circle and a blue square, and the text below them is STRATA140.

All requests used temperature 0 and reasoning effort `none`. Text output
limits were 64 tokens and image output limits were 128 tokens. The second
image request reused the image and prompt cache, so these observations are
functional checks rather than a performance comparison. Fine-detail OCR,
grounding, long-context image use, and sustained throughput were not tested.

The server remains running with the existing
`run-swift-huihui-iq3_xxs.bat` launcher and
`strata-swift-huihui-iq3_xxs.json` config. Structured evidence is in
[verification.json](verification.json); raw setup, build, and request logs
remain in the ignored local log directory.
