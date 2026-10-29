"""The vectorized kernel at every threads-per-block x rows-per-warp config, M = 1 and 8 on the
Llama-3-70B q_proj (8192 x 8192). M = 8 shows what rows per warp do once x is reused.

Usage: uv run python -m bench.block_sweep
"""

import json
from functools import partial
from pathlib import Path

from bench.layers import load
from bench.timing import int4_bytes, measure, measured_peak_gbps, pick_idle_gpu, write_csv

GPU = pick_idle_gpu()

import torch  # noqa: E402

from int4gemv import ops  # noqa: E402
from int4gemv.quant import GROUP_SIZE, quantize  # noqa: E402
from int4gemv.reference import assert_matches_reference, w4a16_reference  # noqa: E402

BATCHES = [1, 8]
LAYER = "llama3_70b_q"
THREADS_PER_BLOCK = [32, 64, 128, 256, 512, 1024]
ROWS_PER_WARP = [1, 2, 4]
RESULTS = Path(__file__).resolve().parents[1] / "outputs"


def main():
    torch.manual_seed(0)
    w = load(LAYER).float()
    n, k = w.shape
    qw, s, z = (t.cuda() for t in quantize(w, GROUP_SIZE))
    peak, peak_meta = measured_peak_gbps(GPU)

    rows, meta = [], {"peak_gbps": peak, "peak": peak_meta}
    for m in BATCHES:
        x = torch.randn(m, k, device="cuda").half()
        ref = w4a16_reference(x, qw, s, z, GROUP_SIZE)
        name = "vectorized"
        kernel = ops.KERNELS[name]
        for tpb in THREADS_PER_BLOCK:
            for rpw in ROWS_PER_WARP:
                config = {"threads_per_block": tpb, "rows_per_warp": rpw}
                info = ops.KERNEL_INFO[name](m, **config)
                if tpb > info["max_threads_per_block"]:
                    print(f"{name} M={m} tpb={tpb:>4} rpw={rpw}: too many registers, skipped")
                    continue
                fn = partial(kernel, x, qw, s, z, GROUP_SIZE, **config)
                assert_matches_reference(fn(), ref)
                result = measure(fn, GPU)
                meta[f"{name}/{m}/{tpb}/{rpw}"] = result.pop("meta")
                gbps = int4_bytes(m, n, k, GROUP_SIZE) / result["time_us"] / 1e3
                rows.append(
                    {
                        "kernel": name,
                        "m": m,
                        **config,
                        **result,
                        "gbps": gbps,
                        "pct_peak": 100 * gbps / peak,
                        "registers": info["registers"],
                        "local_bytes": info["local_bytes"],
                        "warps_per_sm": info["warps_per_sm"],
                    }
                )
                print(
                    f"{name} M={m} tpb={tpb:>4} rpw={rpw}: {result['time_us']:7.1f} µs "
                    f"{gbps:5.0f} GB/s ({100 * gbps / peak:3.0f}%), "
                    f"{info['warps_per_sm']} warps/SM"
                )

    write_csv(RESULTS / "block_sweep.csv", rows)
    (RESULTS / "block_sweep_meta.json").write_text(json.dumps(meta, indent=1) + "\n")


if __name__ == "__main__":
    main()
