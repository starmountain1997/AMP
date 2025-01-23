from typing import Optional, Union
import torch

def rotary_kernel(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor],
    cu_seqlens: Optional[torch.Tensor],
    seqlen: int,
    rotary_dim: int,
    seqlen_ro: int,
    interleaved: bool = False,
    conjugate: bool = False,
) -> torch.Tensor:
    batch, seqlen, nheads, headdim = x.shape
    rotary_dim_half = rotary_dim // 2
    output = torch.empty_like(x)

    if interleaved:
        for b in range(batch):
            for h in range(nheads):
                x0 = x[b, :, h, 0::2]  # Even indices
                x1 = x[b, :, h, 1::2]  # Odd indices
                cos_b = cos[:, :rotary_dim_half]
                sin_b = sin[:, :rotary_dim_half]

                if conjugate:
                    sin_b = -sin_b

                o0 = x0 * cos_b - x1 * sin_b
                o1 = x0 * sin_b + x1 * cos_b

                output[b, :, h, 0::2] = o0
                output[b, :, h, 1::2] = o1
    else:
        for b in range(batch):
            for h in range(nheads):
                for m in range(seqlen):
                    if m >= seqlen_ro:
                        break

                    x0 = x[b, m, h, :rotary_dim_half]
                    x1 = x[b, m, h, rotary_dim_half:rotary_dim]
                    cos_m = cos[m, :rotary_dim_half]
                    sin_m = sin[m, :rotary_dim_half]

                    if conjugate:
                        sin_m = -sin_m

                    o0 = x0 * cos_m - x1 * sin_m
                    o1 = x0 * sin_m + x1 * cos_m

                    output[b, m, h, :rotary_dim_half] = o0
                    output[b, m, h, rotary_dim_half:rotary_dim] = o1

    return output


def apply_rotary_pytorch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved=False,
    inplace=False,
    conjugate=False,
) -> torch.Tensor:
    """
    Arguments:
        x: (batch, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim).
        cos: (seqlen_ro, rotary_dim / 2)
        sin: (seqlen_ro, rotary_dim / 2)
        seqlen_offsets: integer or integer tensor of size (batch,)
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Returns:
        y: (batch, seqlen, nheads, headdim)
    """
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert (
            max_seqlen is not None
        ), "If cu_seqlens is passed in, then max_seqlen must be passed"
        total_seqlen, nheads, headdim = x.shape
        batch_p_1 = cu_seqlens.shape[0]
        batch_p_1 - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim = cos.shape
    assert sin.shape == cos.shape
    rotary_dim *= 2
    assert rotary_dim <= headdim, "rotary_dim must be <= headdim"
    assert headdim <= 256, "Only support headdim <= 256"
    assert seqlen_ro >= seqlen, "seqlen_ro must be >= seqlen"

    assert (
        cos.dtype == sin.dtype
    ), f"cos and sin must have the same dtype, got {cos.dtype} and {sin.dtype}"
    assert (
        x.dtype == cos.dtype
    ), f"Input and cos/sin must have the same dtype, got {x.dtype} and {cos.dtype}"

    cos, sin = cos.contiguous(), sin.contiguous()

    # if isinstance(seqlen_offsets, torch.Tensor):
    #     assert seqlen_offsets.shape == (batch,)
    #     assert seqlen_offsets.dtype in [torch.int32, torch.int64]
    #     seqlen_offsets = seqlen_offsets.contiguous()
    # else:
    #     assert seqlen_offsets + seqlen <= seqlen_ro

    output = rotary_kernel(
        x,
        cos,
        sin,
        seqlen_offsets,
        cu_seqlens,
        seqlen,
        rotary_dim,
        seqlen_ro,
        interleaved,
        conjugate,
    )

    return output