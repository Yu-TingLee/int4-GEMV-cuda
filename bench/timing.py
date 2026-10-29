"""Shared benchmark code: pick an idle GPU, flush L2, time with CUDA events, log nvidia-smi.

We flush L2 by reading a big buffer, not writing one. A write flush (zero_(), like Triton's
do_bench) leaves L2 full of dirty lines that get written back during the next timed call; on an
RTX 4090 that made the vectorized kernel look like 67% of peak instead of 92%.
"""

import csv
import os
import statistics
import subprocess

import torch

FLUSH_FLOATS = 64 * 1024 * 1024  # 256 MB, far larger than any L2 (6 MB on an RTX 3090)


def _smi(query: str, kind: str = "gpu") -> list[list[str]]:
    out = subprocess.run(
        ["nvidia-smi", f"--query-{kind}={query}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [[f.strip() for f in line.split(",")] for line in out.strip().splitlines()]


def _pids_by_uuid() -> dict[str, list[int]]:
    pids: dict[str, list[int]] = {}
    for uuid, pid in _smi("gpu_uuid,pid", kind="compute-apps"):
        pids.setdefault(uuid, []).append(int(pid))
    return pids


def pick_idle_gpu() -> int:
    """Pin this process to an idle GPU (or the least busy one). Call before CUDA starts."""
    gpus = _smi("index,uuid,memory.used")
    busy = _pids_by_uuid()
    idle = [g for g in gpus if g[1] not in busy]
    index, _, _ = idle[0] if idle else min(gpus, key=lambda g: int(g[2]))
    if not idle:
        print(f"warning: no idle GPU; using GPU {index}, results may be contended")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = index
    return int(index)


def gpu_snapshot(index: int) -> dict:
    """Other processes, utilization and clocks of one GPU."""
    fields = "uuid,name,utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,power.draw"
    row = next(r for r in _smi("index," + fields) if int(r[0]) == index)
    snap = dict(zip(fields.split(","), row[1:], strict=True))
    snap["other_pids"] = [p for p in _pids_by_uuid().get(snap["uuid"], []) if p != os.getpid()]
    return snap


def guarded(measure, gpu: int, attempts: int = 3):
    """Retry measure() if another process shows up on the GPU. Returns (result, metadata)."""
    for attempt in range(attempts):
        before = gpu_snapshot(gpu)
        result = measure()
        after = gpu_snapshot(gpu)
        contended = bool(before["other_pids"] or after["other_pids"])
        if not contended or attempt == attempts - 1:
            return result, {"before": before, "after": after, "contended": contended}
        print(f"GPU {gpu} shared by {before['other_pids'] or after['other_pids']}; retrying")


_flush_buf = None


def time_cuda(fn, iters: int = 100, warmup: int = 10) -> list[float]:
    """µs per call of fn(), with L2 flushed before each call."""
    global _flush_buf
    if _flush_buf is None:
        _flush_buf = torch.ones(FLUSH_FLOATS, device="cuda")
    for _ in range(warmup):
        fn()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends, strict=True):
        # The flush also gives the CPU time to queue the launch, so launch latency stays hidden.
        _flush_buf.sum()
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends, strict=True)]


def measure(fn, gpu: int, runs: int = 3) -> dict:
    """Median of the per-run medians over a few separate runs, plus their spread."""
    medians, metas = [], []
    for _ in range(runs):
        times, meta = guarded(lambda: time_cuda(fn), gpu)
        medians.append(statistics.median(times))
        metas.append(meta)
    return {
        "time_us": statistics.median(medians),
        "time_us_min_run": min(medians),
        "time_us_max_run": max(medians),
        "contended": any(m["contended"] for m in metas),
        "meta": metas,
    }


def write_csv(path, rows: list[dict]) -> None:
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def measured_peak_gbps(gpu: int, nbytes: int = 1 << 30) -> tuple[float, dict]:
    """GB/s of a 1 GiB device copy, counting reads and writes. This is our 100%."""
    src = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    times, meta = guarded(lambda: time_cuda(lambda: dst.copy_(src), iters=20), gpu)
    return 2 * nbytes / statistics.median(times) / 1e3, meta


def int4_bytes(m: int, n: int, k: int, group_size: int) -> int:
    """Bytes a W4A16 call has to move: packed weights, scales, zeros, x and y."""
    return n * k // 2 + 2 * n * (k // group_size) * 2 + m * k * 2 + m * n * 2


def fp16_bytes(m: int, n: int, k: int) -> int:
    """Bytes an FP16 call has to move: weights, x and y."""
    return n * k * 2 + m * k * 2 + m * n * 2
