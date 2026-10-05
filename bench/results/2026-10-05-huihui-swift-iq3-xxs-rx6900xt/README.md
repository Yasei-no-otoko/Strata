# Huihui Swift IQ3_XXS installation on RX 6900 XT

On 2026-10-05 (Asia/Tokyo), the local Qwen3.8-Flash-Next Q2_0 and Coder IQ1_M
installations were replaced with Huihui's Swift IQ3_XXS variant. The user
explicitly requested removal of both old models. The new model is running on
`127.0.0.1:8081`, using the existing Strata 0.1.39 Windows HIP engine.

## Model provenance

The weights came from
[huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF/tree/1e1bd8216ba46a590d8168cca08f68de9536e4ac/swift-IQ3_XXS),
revision `1e1bd8216ba46a590d8168cca08f68de9536e4ac`, folder `swift-IQ3_XXS`.
The repository describes these Swift files as derived from UkisAI's Swift 1.5.
The two downloaded files total 75,966,073,120 bytes. Both SHA-256 hashes were
computed over the complete local files and matched the pinned Hugging Face
LFS metadata. Strata's GGUF structure checks also passed.

| File suffix (prefix `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-`) | Bytes | SHA-256 |
| --- | ---: | --- |
| `00001-of-00002.gguf` | 39,785,790,560 | `d815c962f3436e5571f81f874989b89cd25443a15c6b39f9336982b8d1e42d36` |
| `00002-of-00002.gguf` | 36,180,282,560 | `953b8b1ad56dd6885ac724a770e3b6e914e9e65287f61ee853da285914503cd5` |

## Installation and retained settings

Hardware: Ryzen Threadripper 3990X (64 cores, 128 logical processors),
137,306,128,384 bytes of system RAM, and RX 6900 XT (gfx1030, 16 GiB).
AMD driver: `32.0.21045.5002`. The physical GPU was detected by the HIP device
probe and named in the engine startup log.

The source baseline was `087849a5bcd707286979bf95e14439039fb087a6`.
The existing engine was reused; its SHA-256 is
`fe616b0f61b0ce1f18f37a9bd65542f883a64e78384230ffb42db927a59a6df2`.

Setup used its existing Swift family and the verified local files through
`--gguf-dir`, with `--yes --setup --model IQ3_XXS --backend hip --context 131072
--port 8081 --host 127.0.0.1 --vision no --no-start`.
A process-local override of `setup.FAMILIES['swift']` set the tag to
`swift-huihui-` and the model name to `huihui-swift-1.5-abliterated`.
This keeps Huihui's weights separate from the stock Swift download and retains
the recognized `swift-` prefix. No tracked setup or engine source was changed.

The resulting local files are:

- GGUFs: `C:\Dev\Strata-data\models\huihui-swift-IQ3_XXS\`.
- Native pack: `C:\Dev\Strata-data\packs\swift-huihui-iq3_xxs\`.
- Config: `C:\Dev\Strata\strata-swift-huihui-iq3_xxs.json`.
- Launcher: `C:\Dev\Strata\run-swift-huihui-iq3_xxs.bat`.

The pack and tokenizer were generated from the new GGUFs, including their chat
template. Packing reported 96 exact conversions and zero rounded conversions.
The PLE table correctly points to Swift's first shard. The original shared MTP
draft layer was retained because the Swift configuration uses it. Context
131072, KV `int8`, resident KV 32768, speculative decoding 4, and the loopback
port match the previous ordinary Q2_0 installation. Images remain disabled.
The launcher retains the previous behavior of opening the browser.

The engine reported 63 expert-pool workers plus the host thread and
`session is up (engine 0.1.39)`. Setup reported no hipBLASLt tuning table for
gfx1030 with this library version, so dense prompt products use hipBLAS.

## Runtime verification and removal

| Check | Observed result |
| --- | --- |
| `GET /health` | `status: ok`, `loaded: true`, model `huihui-swift-1.5-abliterated-iq3_xxs`, context 131072 |
| `GET /v1/models` | The same model identifier is listed |
| `GET /props` | Strata 0.1.39 |
| `GET /` | HTTP 200, Strata HTML |
| English chat completion | Exact response `Swift ready.`, finish reason `stop` |
| Japanese chat completion | Response `こんにちは。`, finish reason `stop` |
| Responses API | Exact response `Ready.`, status `completed` |

Generation checks used temperature 0, reasoning effort `none`, and a 64-token
output limit. These were real inference requests on the physical AMD GPU.

After the checks passed, both old model directories (`models/Q2_0` and
`models/coder-IQ1_M`), their packs, and their active configs and launchers were
deleted. Their second GGUF shards had shared a hard link. Drive C free space
increased by 99,013,509,120 bytes across cleanup; this is an observed filesystem
delta, not the sum of the two logical model sizes. Small configuration backups
and logs remain under `logs/huihui-swift-20261005/`; no old full model was backed
up there. The shared MTP files remain in use by the new model.

A subsequent health check still reported the Swift model loaded. The only
remaining installed model config and launcher are Swift's, and the listener
was verified on `127.0.0.1:8081`. The server remains running. No Windows reboot
was needed. Evidence is summarized in [verification.json](verification.json).

This run verifies installation and short inference. It does not measure
long-context behavior, sustained throughput, or benchmark quality.
