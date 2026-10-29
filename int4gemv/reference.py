"""The fp32 reference every kernel must match before it is timed."""

import torch

from int4gemv.quant import dequantize


def w4a16_reference(x, qweight, scales, zeros, group_size: int) -> torch.Tensor:
    """Return x [M, K] @ dequant(W).T in fp32, shape [M, N]."""
    return x.float() @ dequantize(qweight, scales, zeros, group_size).T


def assert_matches_reference(y: torch.Tensor, ref: torch.Tensor) -> None:
    """Check a kernel's fp16 output against the fp32 reference.

    The kernels sum in a different order and round to fp16 at the end, so allow ~1% error.
    atol scales with the output's RMS so the check doesn't depend on the weight scale. RMS rather
    than std: std is NaN for a single output and 0 when all outputs are equal.
    """
    assert y.dtype == torch.float16 and y.shape == ref.shape, (y.dtype, y.shape, ref.shape)
    rms = torch.linalg.vector_norm(ref).item() / max(ref.numel(), 1) ** 0.5
    torch.testing.assert_close(y.float().cpu(), ref.cpu(), rtol=1e-2, atol=1e-2 * rms)
