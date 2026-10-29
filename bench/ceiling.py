"""Is the vectorized kernel limited by memory or by its math? Compares it, at M = 1, with its own
loads alone and with torch's fp32 sum over the same bytes.

A device copy would be a bad reference here: its writes sit in L2 until after the timed call.
Usage: uv run python -m bench.ceiling
"""

import json
from functools import partial
from pathlib import Path

from bench.layers import LAYERS, load
from bench.timing import int4_bytes, measure, measured_peak_gbps, pick_idle_gpu, write_csv

GPU = pick_idle_gpu()

import torch  # noqa: E402

from int4gemv import ops  # noqa: E402
from int4gemv.quant import GROUP_SIZE, quantize  # noqa: E402
from int4gemv.reference import assert_matches_reference, w4a16_reference  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "outputs"


def main():
    peak, peak_meta = measured_peak_gbps(GPU)
    rows, meta = [], {"peak_gbps": peak, "peak": peak_meta}
    for shape in LAYERS:
        w = load(shape).float()
        n, k = w.shape
        nbytes = int4_bytes(1, n, k, GROUP_SIZE)
        x = torch.randn(1, k, device="cuda").half()
        qw, s, z = (t.cuda() for t in quantize(w, GROUP_SIZE))
        # load_only and sum compute no GEMV, so only vectorized has a result to check
        assert_matches_reference(
            ops.vectorized(x, qw, s, z, GROUP_SIZE), w4a16_reference(x, qw, s, z, GROUP_SIZE)
        )
        floats = torch.ones(nbytes // 4, device="cuda")
        configs = {
            "sum": floats.sum,
            "load_only": partial(ops.load_only, x, qw, s, z, GROUP_SIZE),
            "vectorized": partial(ops.vectorized, x, qw, s, z, GROUP_SIZE),
        }
        for name, fn in configs.items():
            result = measure(fn, GPU)
            meta[f"{shape}/{name}"] = result.pop("meta")
            gbps = nbytes / result["time_us"] / 1e3
            rows.append(
                {
                    "shape": shape,
                    "n": n,
                    "k": k,
                    "kernel": name,
                    **result,
                    "gbps": gbps,
                    "pct_peak": 100 * gbps / peak,
                }
            )
            print(f"{shape:<16} {name:<16} {result['time_us']:7.1f} µs {100 * gbps / peak:4.0f}%")

    write_csv(RESULTS / "ceiling.csv", rows)
    (RESULTS / "ceiling_meta.json").write_text(json.dumps(meta, indent=1) + "\n")


if __name__ == "__main__":
    main()
