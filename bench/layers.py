"""Real Llama-3 weight matrices for the benchmarks, one layer per shape.

Only the needed tensor is fetched, by HTTP range requests into the safetensors shard, so no full
checkpoint is downloaded. Tensors are cached in data/. Speed doesn't depend on the values, but real
weights make the quantization error and the correctness checks meaningful.
"""

import json
import os
import urllib.request
from pathlib import Path

import torch

# Ungated mirrors of the Meta checkpoints, with identical weights
LLAMA3_8B = "NousResearch/Meta-Llama-3-8B"
LLAMA3_70B = "NousResearch/Meta-Llama-3-70B"

# name: (repo, tensor). Middle layers, since the first and last are often atypical.
LAYERS = {
    "llama3_70b_q": (LLAMA3_70B, "model.layers.40.self_attn.q_proj.weight"),  # 8192 x 8192
    "llama3_8b_q": (LLAMA3_8B, "model.layers.16.self_attn.q_proj.weight"),  # 4096 x 4096
    "llama3_8b_up": (LLAMA3_8B, "model.layers.16.mlp.up_proj.weight"),  # 14336 x 4096
    "llama3_8b_down": (LLAMA3_8B, "model.layers.16.mlp.down_proj.weight"),  # 4096 x 14336, long K
}

CACHE = Path(__file__).resolve().parents[1] / "data"
ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def _get(url: str, start: int | None = None, end: int | None = None) -> bytes:
    """Bytes [start, end) of url, or all of it."""
    request = urllib.request.Request(url)
    if start is not None:
        request.add_header("Range", f"bytes={start}-{end - 1}")
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def load(name: str) -> torch.Tensor:
    """The weight matrix [N, K] of layer `name` in LAYERS, on the CPU in its stored dtype."""
    path = CACHE / f"{name}.pt"
    if path.exists():
        return torch.load(path)

    repo, tensor = LAYERS[name]
    base = f"{ENDPOINT}/{repo}/resolve/main"
    shard = json.loads(_get(f"{base}/model.safetensors.index.json"))["weight_map"][tensor]
    url = f"{base}/{shard}"
    # safetensors: 8-byte little-endian header length, a JSON header, then the raw tensors
    header_len = int.from_bytes(_get(url, 0, 8), "little")
    info = json.loads(_get(url, 8, 8 + header_len))[tensor]
    begin, end = (8 + header_len + offset for offset in info["data_offsets"])
    print(f"fetching {repo} {tensor} ({(end - begin) / 1e6:.0f} MB)")
    raw = bytearray(_get(url, begin, end))
    w = torch.frombuffer(raw, dtype=_DTYPES[info["dtype"]]).reshape(info["shape"]).clone()

    CACHE.mkdir(exist_ok=True)
    torch.save(w, path)
    return w
