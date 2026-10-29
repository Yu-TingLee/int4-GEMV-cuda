"""Round-to-nearest 4-bit quantization in the packed format the kernels read."""

import torch

GROUP_SIZE = 128  # weights per scale and zero, as in AWQ and GPTQ checkpoints
_SHIFTS = torch.arange(0, 32, 4, dtype=torch.int32)


def bits_per_weight(group_size: int = GROUP_SIZE) -> float:
    """4 bits per weight, plus a 16-bit scale and a 16-bit zero per group."""
    return 4 + 2 * 16 / group_size


def quantize(w: torch.Tensor, group_size: int = GROUP_SIZE):
    """Quantize w [N, K] to (qweight [N, K/8] int32, scales [N, K/g] fp16, zeros [N, K/g] fp16).

    Each group's [min, max], widened to include 0, is spread over the 16 levels 0..15, so
    w ≈ (q - z) * s. Including 0 keeps the zero point z in 0..15.
    """
    n, k = w.shape
    assert k % group_size == 0 and group_size % 8 == 0, (k, group_size)
    wg = w.float().reshape(n, k // group_size, group_size)
    w_min = wg.amin(-1, keepdim=True).clamp(max=0)
    w_max = wg.amax(-1, keepdim=True).clamp(min=0)
    # Round to fp16 first so q is computed with the same scale the kernels will see
    scales = ((w_max - w_min) / 15).clamp(min=1e-6).half().float()
    zeros = (-w_min / scales).round().clamp(0, 15)
    q = (wg / scales + zeros).round().clamp(0, 15).to(torch.int32).reshape(n, k)
    return pack(q), scales.squeeze(-1).half(), zeros.squeeze(-1).half()


def pack(q: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit values q [N, K] int into [N, K/8] int32, lowest bits holding the lowest k."""
    n, k = q.shape
    words = (q.to(torch.int64).reshape(n, k // 8, 8) << _SHIFTS.to(q.device, torch.int64)).sum(-1)
    # torch has no uint32, so wrap the top bit around into int32
    return (words - (words >= 2**31) * 2**32).to(torch.int32)


def unpack(qweight: torch.Tensor) -> torch.Tensor:
    """Unpack qweight [N, K/8] int32 into 4-bit values [N, K] int32."""
    n, words = qweight.shape
    q = (qweight.unsqueeze(-1) >> _SHIFTS.to(qweight.device)) & 0xF
    return q.reshape(n, words * 8)


def dequantize(qweight, scales, zeros, group_size: int) -> torch.Tensor:
    """Return the fp32 weights [N, K] that the packed format represents."""
    q = unpack(qweight).float()
    n, k = q.shape
    qg = q.reshape(n, k // group_size, group_size)
    w = (qg - zeros.float().unsqueeze(-1)) * scales.float().unsqueeze(-1)
    return w.reshape(n, k)
