"""Draws the figures in plots/ from whatever CSVs are in outputs/.

Text is set with LaTeX in Computer Modern when `latex` is on PATH, else in a serif fallback.
Usage: uv run python -m bench.plot
"""

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from int4gemv.quant import GROUP_SIZE, bits_per_weight  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
PLOTS = ROOT / "plots"

USETEX = shutil.which("latex") is not None
TIMES = r"$\times$"
MICRO = r"$\mu$"

# seaborn's colorblind palette (passes a CVD check); gray for the cuBLAS baseline
BLUE, ORANGE, GREEN, RED = "#0173b2", "#de8f05", "#029e73", "#d55e00"
GRAY = "#8c8c8c"
INK, INK_2 = "#222222", "#5a5a5a"
DASHED = (0, (3, 2))

KERNELS = ["cublas_fp16", "naive", "coalesced", "vectorized"]
KERNEL_COLORS = {"cublas_fp16": GRAY, "naive": BLUE, "coalesced": ORANGE, "vectorized": GREEN}
SHAPES = ["llama3_70b_q", "llama3_8b_q", "llama3_8b_up", "llama3_8b_down"]
SHAPE_COLORS = [BLUE, ORANGE, GREEN, RED]
LAYER_NAMES = {
    "llama3_70b_q": "70B Q Proj",  # Llama-3-70B; the captions say so
    "llama3_8b_q": "8B Q Proj",
    "llama3_8b_up": "8B MLP Up",
    "llama3_8b_down": "8B MLP Down",
}
LAYER_SIZES = {
    "llama3_70b_q": (8192, 8192),
    "llama3_8b_q": (4096, 4096),
    "llama3_8b_up": (14336, 4096),
    "llama3_8b_down": (4096, 14336),
}
BATCHES = [1, 2, 4, 8, 16, 32]

IDEAL_SPEEDUP = 16 / bits_per_weight(GROUP_SIZE)  # FP16 weight bytes ÷ INT4 weight bytes
# Spec values; only the memory roof is measured
# GPU name: (FP32 TFLOP/s on CUDA cores, FP16 tensor TFLOP/s with FP32 accumulate), dense
COMPUTE_ROOFS = {
    "NVIDIA GeForce RTX 3090": (35.6, 71.0),  # 10496 cores x 1.695 GHz x 2
    "NVIDIA GeForce RTX 4090": (82.6, 165.2),  # 16384 cores x 2.52 GHz x 2
}

SINGLE = (3.4, 2.2)  # one paper column, inches
DOUBLE = (7.0, 1.9)


def tex(text: str) -> str:
    """Escape the characters LaTeX treats specially in plain text."""
    return text.replace("_", r"\_").replace("%", r"\%") if USETEX else text


def kernel_label(kernel: str) -> str:
    return "cuBLAS FP16" if kernel == "cublas_fp16" else kernel


def layer_label(shape: str) -> str:
    n, k = LAYER_SIZES[shape]
    return f"{LAYER_NAMES[shape]}, {n}{TIMES}{k}"


def setup():
    sns.set_theme(context="paper", style="ticks")
    plt.rcParams.update(
        {
            "text.usetex": USETEX,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"] if USETEX else ["STIXGeneral"],
            "mathtext.fontset": "cm" if USETEX else "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "axes.linewidth": 0.6,
            "axes.titlepad": 4,
            "lines.linewidth": 1.3,
            "lines.markersize": 4,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "savefig.facecolor": "white",
        }
    )


def read(name: str) -> list[dict] | None:
    path = OUTPUTS / name
    if not path.exists():
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def lookup(rows, **match) -> dict:
    return next(r for r in rows if all(r[k] == str(v) for k, v in match.items()))


def save(fig, name: str):
    sns.despine(fig)
    PLOTS.mkdir(exist_ok=True)
    fig.savefig(PLOTS / name, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote plots/{name}")


def bars(ax, labels, values, colors):
    """Bars with their values on top. labels=None leaves the x axis bare for a shared legend."""
    for i, (v, color) in enumerate(zip(values, colors, strict=True)):
        label = labels[i] if labels else None
        ax.bar(i, v, width=0.68, color=color, edgecolor="white", linewidth=0.6, label=label)
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5, color=INK_2)
    ax.tick_params(axis="x", length=0)
    ax.set_xticks(range(len(values)), labels if labels else [""] * len(values))


def kernel_panels(rows, column: str, ylabel: str, name: str, roof: float | None = None):
    """One panel per layer, one bar per kernel at batch 1, kernels named in a shared legend."""
    fig, axes = plt.subplots(1, len(SHAPES), figsize=DOUBLE, layout="constrained")
    for ax, shape in zip(axes, SHAPES, strict=True):
        values = [float(lookup(rows, shape=shape, m=1, kernel=k)[column]) for k in KERNELS]
        bars(ax, None, values, [KERNEL_COLORS[k] for k in KERNELS])
        ax.set_xticks([])
        ax.set_title(layer_label(shape))
        if roof:
            ax.axhline(roof, color=INK_2, linewidth=0.7, linestyle=DASHED)
            ax.set_ylim(0, roof * 1.12)
        else:
            ax.set_ylim(0, max(values) * 1.12)
    axes[0].set_ylabel(ylabel)
    handles = [plt.Rectangle((0, 0), 1, 1, color=KERNEL_COLORS[k]) for k in KERNELS]
    fig.legend(handles, [kernel_label(k) for k in KERNELS], loc="outside upper center", ncol=4)
    save(fig, name)


def ideal_line(ax, x: float, dy: float):
    """Dashed line at the speedup limit set by the byte savings, labelled ending at x."""
    ax.axhline(IDEAL_SPEEDUP, color=INK_2, linewidth=0.7, linestyle=DASHED)
    ax.text(
        x,
        IDEAL_SPEEDUP + dy,
        f"ideal {IDEAL_SPEEDUP:.2f}{TIMES}",
        ha="right",
        va="bottom",
        fontsize=6.5,
        color=INK_2,
    )


def log2_batch_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(BATCHES, [str(m) for m in BATCHES])
    ax.minorticks_off()
    ax.set_xlabel("Batch size $M$")


def plot_time(rows):
    """Time per call of every kernel at batch 1."""
    kernel_panels(rows, "time_us", f"Time ({MICRO}s)", "time.png")


def plot_speedup(rows):
    """Speedup of each kernel over cuBLAS FP16 at batch 1."""
    fig, ax = plt.subplots(figsize=SINGLE, layout="constrained")
    width = 0.26
    for i, kernel in enumerate(KERNELS[1:]):
        xs = [j + (i - 1) * width for j in range(len(SHAPES))]
        ys = [float(lookup(rows, shape=s, m=1, kernel=kernel)["speedup_vs_cublas"]) for s in SHAPES]
        ax.bar(xs, ys, width=width, label=kernel, color=KERNEL_COLORS[kernel], edgecolor="white")
        if kernel == "vectorized":
            for x, y in zip(xs, ys, strict=True):
                ax.text(x, y, f"{y:.1f}{TIMES}", ha="center", va="bottom", fontsize=6.5)
    ideal_line(ax, len(SHAPES) - 0.5, 0.05)
    ax.set_xticks(range(len(SHAPES)), [LAYER_NAMES[s] for s in SHAPES])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Speedup over cuBLAS FP16")
    ax.set_xlim(-0.55, len(SHAPES) - 0.45)
    ax.set_ylim(0, 4.6)
    ax.legend(loc="upper left", ncol=3, handlelength=1.2, columnspacing=0.8)
    save(fig, "speedup.png")


def plot_batch(rows):
    """Speedup of vectorized over cuBLAS FP16 as the batch grows."""
    fig, ax = plt.subplots(figsize=SINGLE, layout="constrained")
    for shape, color, marker in zip(SHAPES, SHAPE_COLORS, "osD^", strict=True):
        ys = [
            float(lookup(rows, shape=shape, m=m, kernel="vectorized")["speedup_vs_cublas"])
            for m in BATCHES
        ]
        ax.plot(BATCHES, ys, color=color, marker=marker, label=layer_label(shape))
    ideal_line(ax, BATCHES[-1], 0.06)
    log2_batch_axis(ax)
    ax.set_ylabel("Speedup over cuBLAS FP16")
    ax.set_ylim(0, 4.2)
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.88), fontsize=6, handlelength=1.6)
    save(fig, "batch.png")


def plot_bandwidth(rows):
    """Effective bandwidth of every kernel at batch 1, as % of the measured peak."""
    kernel_panels(rows, "pct_peak", tex("% of peak bandwidth"), "bandwidth.png", roof=100)


def plot_roofline(rows, peak_gbps: float, gpu: str):
    """Achieved TFLOP/s against FLOPs per byte read, on the 70B q_proj for M = 1 ... 32."""
    shape = "llama3_70b_q"
    n, k = LAYER_SIZES[shape]
    fig, ax = plt.subplots(figsize=(3.4, 2.4), layout="constrained")
    lo, hi = 0.5, 1000
    fp32_tflops, tensor_tflops = COMPUTE_ROOFS[gpu]
    tflops_per_ai = peak_gbps / 1e3
    ridge_fp32 = fp32_tflops / tflops_per_ai
    ridge_tensor = tensor_tflops / tflops_per_ai
    ax.plot(
        [lo, ridge_fp32, hi],
        [lo * tflops_per_ai, fp32_tflops, fp32_tflops],
        color=INK_2,
        linewidth=0.9,
    )
    ax.plot(
        [ridge_fp32, ridge_tensor, hi],
        [fp32_tflops, tensor_tflops, tensor_tflops],
        color=INK_2,
        linewidth=0.9,
        linestyle=DASHED,
    )
    ax.text(
        hi * 0.8,
        fp32_tflops * 0.8,
        "FP32 cores",
        ha="right",
        va="top",
        fontsize=6.5,
        color=INK_2,
    )
    ax.text(
        hi * 0.8,
        tensor_tflops * 1.15,
        "FP16 tensor cores",
        ha="right",
        va="bottom",
        fontsize=6.5,
        color=INK_2,
    )
    for kernel, marker in [("cublas_fp16", "s"), ("vectorized", "o")]:
        xs, ys = [], []
        for m in BATCHES:
            row = lookup(rows, shape=shape, m=m, kernel=kernel)
            seconds = float(row["time_us"]) * 1e-6
            flops = 2 * m * n * k
            xs.append(flops / (float(row["gbps"]) * 1e9 * seconds))
            ys.append(flops / seconds / 1e12)
        ax.plot(xs, ys, color=KERNEL_COLORS[kernel], marker=marker, label=kernel_label(kernel))
        # Only the INT4 points: cuBLAS sits on the roof, where labels would collide
        if kernel == "vectorized":
            for m, x, y in zip(BATCHES, xs, ys, strict=True):
                if m in (1, 4, 16, 32):
                    ax.annotate(
                        f"$M={m}$",
                        (x, y),
                        xytext=(0, -10),
                        textcoords="offset points",
                        fontsize=6,
                        color=INK_2,
                        ha="center",
                    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Plain decade labels and no minor ticks
    ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
    ax.set_yticks([1, 10, 100], ["1", "10", "100"])
    ax.minorticks_off()
    ax.set_xlim(lo, hi)
    ax.set_ylim(0.3, 500)
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("Performance (TFLOP/s)")
    ax.legend(loc="upper left")
    save(fig, "roofline.png")


def plot_profile(rows):
    """Measured achieved occupancy and DRAM throughput (ncu) on the 70B Q Proj: each kernel at
    batch 1, and vectorized as M grows."""
    shape = "llama3_70b_q"
    series = [
        ("achieved_occupancy_pct", "Achieved occupancy", BLUE, "o"),
        ("dram_throughput_pct", "DRAM throughput", ORANGE, "s"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.6), layout="constrained")

    kernels = KERNELS[1:]
    width = 0.36
    for i, (column, _, color, _) in enumerate(series):
        xs = [j + (i - 0.5) * width for j in range(len(kernels))]
        ys = [float(lookup(rows, shape=shape, kernel=k, m=1)[column]) for k in kernels]
        axes[0].bar(xs, ys, width=width, color=color, edgecolor="white", linewidth=0.6)
        for x, y in zip(xs, ys, strict=True):
            axes[0].text(x, y, f"{y:.0f}", ha="center", va="bottom", fontsize=6.5, color=INK_2)
    axes[0].set_xticks(range(len(kernels)), kernels)
    axes[0].tick_params(axis="x", length=0)
    axes[0].set_title("Batch 1")

    for column, _, color, marker in series:
        ys = [float(lookup(rows, shape=shape, kernel="vectorized", m=m)[column]) for m in BATCHES]
        axes[1].plot(BATCHES, ys, color=color, marker=marker)
    log2_batch_axis(axes[1])
    axes[1].set_title("vectorized")

    for ax in axes:
        ax.set_ylim(0, 100)
        ax.set_ylabel(tex("% of peak"))
    handles = [plt.Line2D([], [], color=c, marker=m) for _, _, c, m in series]
    fig.legend(handles, [label for _, label, _, _ in series], loc="outside upper center", ncol=2)
    save(fig, "profile.png")


def plot_ceiling(rows):
    """vectorized against its own loads alone and torch's FP32 sum over the same bytes."""
    names = {"sum": "torch sum", "load_only": "loads only", "vectorized": "vectorized"}
    fig, axes = plt.subplots(1, len(SHAPES), figsize=DOUBLE, sharey=True, layout="constrained")
    for ax, shape in zip(axes, SHAPES, strict=True):
        pcts = [float(lookup(rows, shape=shape, kernel=k)["pct_peak"]) for k in names]
        bars(ax, list(names.values()), pcts, [GRAY, GRAY, GREEN])
        ax.axhline(100, color=INK_2, linewidth=0.7, linestyle=DASHED)
        ax.set_title(layer_label(shape))
        ax.set_ylim(0, 112)
    axes[0].set_ylabel(tex("% of peak bandwidth"))
    for ax in axes[1:]:
        ax.tick_params(axis="y", left=False)  # shared axis; despine drops the spine, not the ticks
    save(fig, "ceiling.png")


def plot_block_sweep(rows):
    """vectorized at every threads-per-block x rows-per-warp config, one panel per batch size."""
    batches = sorted({int(r.get("m", 1)) for r in rows})  # older CSVs have M = 1 only
    fig, axes = plt.subplots(
        1,
        len(batches),
        figsize=(SINGLE[0] * len(batches), SINGLE[1]),
        sharey=True,
        layout="constrained",
        squeeze=False,
    )
    for ax, m in zip(axes[0], batches, strict=True):
        panel = [r for r in rows if int(r.get("m", 1)) == m]
        for rpw, color, marker in zip(["1", "2", "4"], SHAPE_COLORS[:3], "osD", strict=True):
            pts = sorted(
                (int(r["threads_per_block"]), float(r["pct_peak"]))
                for r in panel
                if r["rows_per_warp"] == rpw
            )
            if not pts:
                continue
            xs, ys = zip(*pts, strict=True)
            label = f"{rpw} row{'s' if rpw != '1' else ''}/warp"
            ax.plot(xs, ys, color=color, marker=marker, label=label)
        # Configs that ran out of registers leave some series short
        ticks = sorted({int(r["threads_per_block"]) for r in panel})
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("Threads per block")
        if len(batches) > 1:
            ax.set_title(f"M = {m}")
    axes[0][0].set_ylabel(tex("% of peak bandwidth"))
    axes[0][0].set_ylim(0, 105)
    axes[0][0].legend(loc="lower right")
    for ax in axes[0][1:]:
        ax.tick_params(axis="y", left=False)  # shared axis; despine drops the spine, not the ticks
    save(fig, "block_sweep.png")


def main():
    setup()
    if sweep := read("sweep.csv"):
        meta = json.loads((OUTPUTS / "sweep_meta.json").read_text())
        peak, gpu = meta["peak_gbps"], meta["peak"]["before"]["name"]
        plot_time(sweep)
        plot_speedup(sweep)
        plot_batch(sweep)
        plot_bandwidth(sweep)
        plot_roofline(sweep, peak, gpu)
    if ceiling := read("ceiling.csv"):
        plot_ceiling(ceiling)
    if block := read("block_sweep.csv"):
        plot_block_sweep(block)
    if profile := read("profile.csv"):
        plot_profile(profile)


if __name__ == "__main__":
    main()
