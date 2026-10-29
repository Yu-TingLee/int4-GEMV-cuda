"""Hardware counters for each kernel from Nsight Compute: achieved occupancy, DRAM throughput,
sectors per global load request and L1 hit rate. Writes outputs/profile.csv.

Runs itself under ncu, which launches each kernel once per config in a fixed order; ncu's
results come back in that order. Clocks are left to the GPU (--clock-control none), as in the
timing benchmarks; nvidia-smi snapshots before and after go to outputs/profile_meta.json.
Needs profiler access (NVreg_RestrictProfilingToAdminUsers=0, or root).
Usage: uv run python -m bench.profile
"""

import csv
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from bench.layers import LAYERS, load
from bench.timing import gpu_snapshot, pick_idle_gpu, write_csv
from int4gemv.quant import GROUP_SIZE

OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
BATCHES = [1, 2, 4, 8, 16, 32]
KERNEL_REGEX = "regex:naive_kernel|coalesced_kernel|vectorized_kernel"

# column name: ncu metric
METRICS = {
    "time_us": "gpu__time_duration.sum",
    "achieved_occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "dram_throughput_pct": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_read_bytes": "dram__bytes_read.sum",
    "ld_sectors": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "ld_requests": "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1_hit_rate_pct": "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct",
}


def configs() -> list[dict]:
    """Every (layer, kernel, M) to profile, in launch order."""
    out = [
        {"shape": shape, "kernel": kernel, "m": 1}
        for shape in LAYERS
        for kernel in ("naive", "coalesced", "vectorized")
    ]
    out += [{"shape": "llama3_70b_q", "kernel": "vectorized", "m": m} for m in BATCHES[1:]]
    return out


def workload():
    """Launch each config's kernel exactly once. Correctness is checked by the sweep."""
    import torch

    from int4gemv import ops
    from int4gemv.quant import quantize

    torch.manual_seed(0)
    weights = {}
    for c in configs():
        if c["shape"] not in weights:
            weights[c["shape"]] = [t.cuda() for t in quantize(load(c["shape"]).float(), GROUP_SIZE)]
        qw, s, z = weights[c["shape"]]
        x = torch.randn(c["m"], qw.shape[1] * 8, device="cuda").half()
        ops.KERNELS[c["kernel"]](x, qw, s, z, GROUP_SIZE)
    torch.cuda.synchronize()


def main():
    gpu = pick_idle_gpu()  # sets CUDA_VISIBLE_DEVICES, which the ncu child inherits
    ncu = shutil.which("ncu")
    assert ncu, "ncu not on PATH"
    cmd = [
        ncu,
        "--csv",
        "--page=raw",
        "--clock-control=none",
        "--cache-control=all",  # flush caches before every profiled launch
        "--kernel-name",
        KERNEL_REGEX,
        "--metrics",
        ",".join(METRICS.values()),
        sys.executable,
        "-m",
        "bench.profile",
        "--workload",
    ]
    before = gpu_snapshot(gpu)
    result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
    meta = {"before": before, "after": gpu_snapshot(gpu), "ncu": ncu}
    if result.returncode != 0 or "ERR_NVGPUCTRPERM" in result.stdout:
        sys.exit(f"ncu failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
    # ncu prints the program's output and its own log lines before the CSV header
    text = result.stdout[result.stdout.index('"ID"') :]
    rows = list(csv.DictReader(io.StringIO(text)))[1:]  # the first row holds units
    expected = configs()
    assert len(rows) == len(expected), (len(rows), len(expected))

    out = []
    for config, row in zip(expected, rows, strict=True):
        assert config["kernel"] in row["Kernel Name"], (config, row["Kernel Name"])
        values = {col: float(row[metric].replace(",", "")) for col, metric in METRICS.items()}
        values["time_us"] /= 1e3  # ncu reports ns
        values["sectors_per_request"] = values["ld_sectors"] / values["ld_requests"]
        out.append({**config, **values})
        print(
            f"{config['shape']:<14} {config['kernel']:<11} M={config['m']:<3}"
            f" occupancy {values['achieved_occupancy_pct']:5.1f}%"
            f"  DRAM {values['dram_throughput_pct']:5.1f}%"
            f"  sectors/request {values['sectors_per_request']:5.2f}"
            f"  L1 hit {values['l1_hit_rate_pct']:5.1f}%"
        )
    write_csv(OUTPUTS / "profile.csv", out)
    (OUTPUTS / "profile_meta.json").write_text(json.dumps(meta, indent=1) + "\n")


if __name__ == "__main__":
    if "--workload" in sys.argv:
        workload()
    else:
        main()
