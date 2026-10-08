#!/usr/bin/env python3
"""Record a bounded engine run and Windows process/available-memory observations.

Usage: python tools/measure_run.py --out DIR [--timeout 900] -- RUNNER [engine args]
The child inherits the current runtime-library PATH. No model or config is changed.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


class ProcessMemory(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
        (name, ctypes.c_size_t) for name in (
            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
            "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage",
            "PeakPagefileUsage", "PrivateUsage")]


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
        (name, ctypes.c_ulonglong) for name in (
            "total_physical", "available_physical", "total_pagefile", "available_pagefile",
            "total_virtual", "available_virtual", "available_extended_virtual")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        ap.error("an engine command is required after --")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "summary.json").exists():
        ap.error("output already contains summary.json; use a new case directory")
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "command": command,
              "cwd": str(Path.cwd()), "timeout_s": args.timeout, "samples": []}
    get_process = get_memory = None
    if os.name == "nt":
        get_process = ctypes.WinDLL("psapi").GetProcessMemoryInfo
        get_process.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessMemory), ctypes.c_ulong]
        get_process.restype = ctypes.c_int
        get_memory = ctypes.WinDLL("kernel32").GlobalMemoryStatusEx
        get_memory.argtypes = [ctypes.POINTER(MemoryStatus)]
        get_memory.restype = ctypes.c_int
    start = time.perf_counter()
    with (args.out / "stdout.txt").open("w", encoding="utf-8") as out, \
            (args.out / "stderr.txt").open("w", encoding="utf-8") as err:
        proc = subprocess.Popen(command, stdout=out, stderr=err)
        report["pid"] = proc.pid
        try:
            while True:
                elapsed = time.perf_counter() - start
                sample = {"elapsed_s": round(elapsed, 3)}
                if get_process:
                    pm = ProcessMemory(); pm.cb = ctypes.sizeof(pm)
                    if get_process(int(proc._handle), ctypes.byref(pm), pm.cb):
                        sample.update(working_set_bytes=pm.WorkingSetSize,
                                      peak_working_set_bytes=pm.PeakWorkingSetSize,
                                      private_bytes=pm.PrivateUsage,
                                      peak_pagefile_bytes=pm.PeakPagefileUsage)
                    ms = MemoryStatus(); ms.length = ctypes.sizeof(ms)
                    if get_memory(ctypes.byref(ms)):
                        sample.update(system_available_physical_bytes=ms.available_physical,
                                      system_total_physical_bytes=ms.total_physical)
                report["samples"].append(sample)
                if proc.poll() is not None:
                    break
                if elapsed > args.timeout:
                    report["timed_out"] = True
                    proc.kill()
                    proc.wait(timeout=10)
                    break
                time.sleep(0.2)
        except BaseException:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
            raise
        finally:
            report.update(exit_code=proc.returncode, wall_s=round(time.perf_counter() - start, 3))
            for key in ("peak_working_set_bytes", "private_bytes", "peak_pagefile_bytes"):
                values = [s[key] for s in report["samples"] if key in s]
                if values:
                    report[key] = max(values)
            available = [s["system_available_physical_bytes"] for s in report["samples"]
                         if "system_available_physical_bytes" in s]
            if available:
                report["min_system_available_physical_bytes"] = min(available)
            (args.out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2), flush=True)
    return 124 if report.get("timed_out") else proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
