"""Every kernel on every real layer for M = 1 ... 32. Writes outputs/sweep.csv.

Usage: uv run python -m bench.sweep
"""

import json
from functools import partial
from pathlib import Path

from bench.layers import LAYERS, load
from bench.timing import (
    fp16_bytes,
    int4_bytes,
    measure,
    measured_peak_gbps,
    pick_idle_gpu,
    write_csv,
)

GPU = pick_idle_gpu()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from int4gemv import ops  # noqa: E402
from int4gemv.quant import GROUP_SIZE, quantize  # noqa: E402
from int4gemv.reference import assert_matches_reference, w4a16_reference  # noqa: E402

BATCHES = [1, 2, 4, 8, 16, 32]
RESULTS = Path(__file__).resolve().parents[1] / "outputs"


def main():
    peak, peak_meta = measured_peak_gbps(GPU)
    print(f"measured peak {peak:.0f} GB/s")
    rows, meta = [], {"peak_gbps": peak, "peak": peak_meta, "torch": torch.__version__}

    for shape in LAYERS:
        torch.manual_seed(0)
        w = load(shape).float()
        n, k = w.shape
        w16 = w.half().cuda()
        qw, s, z = (t.cuda() for t in quantize(w, GROUP_SIZE))

        for m in BATCHES:
            x = torch.randn(m, k, device="cuda").half()
            ref = w4a16_reference(x, qw, s, z, GROUP_SIZE)
            y16 = F.linear(x, w16).float()

            # name: (fn, bytes moved)
            configs = {"cublas_fp16": (partial(F.linear, x, w16), fp16_bytes(m, n, k))}
            for name, kernel in ops.KERNELS.items():
                configs[name] = (
                    partial(kernel, x, qw, s, z, GROUP_SIZE),
                    int4_bytes(m, n, k, GROUP_SIZE),
                )

            group = []
            for name, (fn, nbytes) in configs.items():
                # Error of the INT4 output against the FP16 one: what quantization costs
                rel_err = None
                if name in ops.KERNELS:
                    y = fn()
                    assert_matches_reference(y, ref)
                    rel_err = ((y.float() - y16).norm() / y16.norm()).item()
                result = measure(fn, GPU)
                meta[f"{shape}/{m}/{name}"] = result.pop("meta")
                info = ops.KERNEL_INFO[name](m) if name in ops.KERNEL_INFO else {}
                gbps = nbytes / result["time_us"] / 1e3
                group.append(
                    {
                        "shape": shape,
                        "n": n,
                        "k": k,
                        "m": m,
                        "kernel": name,
                        **result,
                        "gbps": gbps,
                        "pct_peak": 100 * gbps / peak,
                        "registers": info.get("registers"),
                        "local_bytes": info.get("local_bytes"),
                        "warps_per_sm": info.get("warps_per_sm"),
                        "rel_err_vs_fp16": rel_err,
                    }
                )

            by_name = {r["kernel"]: r for r in group}
            for r in group:
                r["speedup_vs_cublas"] = by_name["cublas_fp16"]["time_us"] / r["time_us"]
            best = by_name["vectorized"]
            print(
                f"{shape:<16} M={m:<3} cuBLAS {by_name['cublas_fp16']['time_us']:8.1f} µs  "
                f"vectorized {best['time_us']:8.1f} µs ({best['pct_peak']:3.0f}% peak)  "
                f"speedup {best['speedup_vs_cublas']:.2f}x"
            )
            rows += group

    write_csv(RESULTS / "sweep.csv", rows)
    (RESULTS / "sweep_meta.json").write_text(json.dumps(meta, indent=1) + "\n")


if __name__ == "__main__":
    main()
