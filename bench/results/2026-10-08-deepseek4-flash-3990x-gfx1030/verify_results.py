"""Verify the exact-token runs and regenerate their compact result table.

Run with Python 3 from any working directory. Raw records remain unchanged.
The model's generated text can differ between float and quantized CPU activations.
"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT.parent / "2026-10-08-deepseek4-flash-windows-hip/short-128-input/prompt-ids.txt"
CASES = ("final128-baseline", "final128-hip", "final128-native32")


def match(pattern, text):
    found = re.search(pattern, text, re.MULTILINE)
    if found is None:
        raise ValueError(f"Missing measurement: {pattern}")
    return found


def main():
    prompt = [int(token) for token in FIXTURE.read_text().strip().split(",")]
    if len(prompt) != 128:
        raise ValueError("Fixture must have exactly 128 input token IDs")
    results = {
        "prompt_fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        "input_tokens": len(prompt),
        "cases": {},
    }
    baseline_ids = None
    for name in CASES:
        case = ROOT / name
        summary = json.loads((case / "summary.json").read_text())
        stdout = (case / "stdout.txt").read_text(encoding="utf-8")
        stderr = (case / "stderr.txt").read_text(encoding="utf-8")
        if summary["exit_code"] != 0 or summary.get("timed_out"):
            raise ValueError(f"{name}: did not finish successfully")
        command = summary["command"]
        supplied = [int(token) for token in command[command.index("--prompt-ids") + 1].split(",")]
        if supplied != prompt or command[command.index("--n-predict") + 1] != "128":
            raise ValueError(f"{name}: benchmark command changed the requested tokens")
        pp = match(r"^prompt: (\d+) tokens in ([\d.]+) s \(([\d.]+) tok/s\)", stdout)
        tg = match(r"^decode: (\d+) tokens in ([\d.]+) s \(([\d.]+) tok/s\)", stdout)
        ids = [int(token) for token in match(r"^generated ids:([^\r\n]+)", stdout).group(1).split()]
        if int(pp.group(1)) != 128 or int(tg.group(1)) != 128 or len(ids) != 128:
            raise ValueError(f"{name}: did not actually process 128 and generate 128 tokens")
        if "device: hip" not in stderr or "gfx1030" not in stderr:
            raise ValueError(f"{name}: missing real HIP gfx1030 device evidence")
        if baseline_ids is None:
            baseline_ids = ids
        row = {
            "executable_sha256": summary["executable_sha256"],
            "prompt_seconds": float(pp.group(2)),
            "prompt_tokens_per_second": float(pp.group(3)),
            "decode_seconds": float(tg.group(2)),
            "decode_tokens_per_second": float(tg.group(3)),
            "wall_seconds_including_load_and_exit": summary["wall_s"],
            "peak_process_working_set_gib": summary["peak_working_set_bytes"] / 2**30,
            "minimum_system_available_physical_gib": summary["min_system_available_physical_bytes"] / 2**30,
            "generated_ids_equal_baseline": ids == baseline_ids,
            "generated_ids_sha256": hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest(),
            "runtime_environment": summary["runtime_environment"],
        }
        for label, pattern in (
            ("cpu_expert_evaluations", r"^  CPU expert evaluations: (\d+)"),
            ("native_cpu_expert_evaluations", r"^  native quantized CPU expert evaluations: (\d+)"),
            ("staged_expert_h2d_gib", r"^  staged expert H2D: ([\d.]+) GiB"),
        ):
            value = re.search(pattern, stderr, re.MULTILINE)
            if value:
                row[label] = float(value.group(1)) if label.endswith("gib") else int(value.group(1))
        results["cases"][name] = row
    native = results["cases"]["final128-native32"]
    if native.get("native_cpu_expert_evaluations") != 255 * 43 * 6:
        raise ValueError("Native run did not evaluate all six routed experts of all 43 layers")
    if native.get("staged_expert_h2d_gib") != 0:
        raise ValueError("Native run unexpectedly staged routed experts to the GPU")
    (ROOT / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("| Build | Prompt tok/s | Decode tok/s | Process wall s | Peak process GiB | Min free RAM GiB |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name, row in results["cases"].items():
        print(f"| {name} | {row['prompt_tokens_per_second']:.2f} | {row['decode_tokens_per_second']:.2f} | "
              f"{row['wall_seconds_including_load_and_exit']:.3f} | {row['peak_process_working_set_gib']:.2f} | "
              f"{row['minimum_system_available_physical_gib']:.2f} |")
    print("Verified: identical 128-token input; 128 generated tokens per run; real HIP; 65,790 native CPU expert evaluations.")


if __name__ == "__main__":
    main()
