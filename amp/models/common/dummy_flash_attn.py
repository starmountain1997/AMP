from typing import Optional, Union

import torch
from loguru import logger

# https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/rotary.py


def apply_rotary_embedding(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
    seqlen_ro: int,
) -> torch.Tensor:
    """
    Apply rotary embedding to the input tensor x.

    Args:
        x (torch.Tensor): Input tensor of shape [batch, seqlen, nheads, headdim].
        cos (torch.Tensor): Cosine tensor of shape [seqlen_ro, rotary_dim//2].
        sin (torch.Tensor): Sine tensor of shape [seqlen_ro, rotary_dim//2].
        rotary_dim (int): Number of dimensions to apply rotary embedding to.
        seqlen_ro (int): Sequence length up to which rotary embedding is applied.
        mask (torch.Tensor, optional): Mask tensor of shape [batch, seqlen],
                                       where 1 indicates valid positions and 0 indicates padding.

    Returns:
        torch.Tensor: Tensor with rotary embedding applied, same shape as x.
    """
    # Validate dimensions
    assert x.dim() == 4, "Input tensor x must be 4-dimensional [batch, seqlen, nheads, headdim]"
    batch, seqlen, nheads, headdim = x.shape
    assert headdim >= rotary_dim, "headdim must be greater than or equal to rotary_dim"
    assert rotary_dim % 2 == 0, "rotary_dim must be divisible by 2"

    # Adjust seqlen_ro if it exceeds the actual sequence length
    seqlen_ro = min(seqlen_ro, seqlen)

    # Slice the tensor to apply rotary embedding only up to seqlen_ro
    x_rot = x[:, :seqlen_ro, :, :rotary_dim]

    # Split the rotary dimensions into two halves
    x1, x2 = x_rot[..., :rotary_dim//2], x_rot[..., rotary_dim//2:]

    # Prepare cos and sin for broadcasting
    # cos and sin should have shape [seqlen_ro, rotary_dim//2]
    # Reshape to [1, seqlen_ro, 1, rotary_dim//2] for broadcasting
    cos = cos[:seqlen_ro, :rotary_dim//2].unsqueeze(0).unsqueeze(2)  # [1, seqlen_ro, 1, rotary_dim//2]
    sin = sin[:seqlen_ro, :rotary_dim//2].unsqueeze(0).unsqueeze(2)  # [1, seqlen_ro, 1, rotary_dim//2]

    # Apply rotation
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos

    # Concatenate the rotated parts
    x_rotated = torch.cat([o1, o2], dim=-1)

    # Create a copy of x to avoid in-place modifications if necessary
    x_out = x.clone()


    # Replace the rotated part in the output tensor
    x_out[:, :seqlen_ro, :, :rotary_dim] = x_rotated

    return x_out

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
        batch = batch_p_1 - 1
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
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        assert seqlen_offsets.dtype in [torch.int32, torch.int64]
        seqlen_offsets = seqlen_offsets.contiguous()
    else:
        assert seqlen_offsets + seqlen <= seqlen_ro

    out=apply_rotary_embedding(x, cos, sin, rotary_dim, seqlen_ro)

    
    return out


class ApplyRotaryEmb(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        cos,
        sin,
        interleaved=False,
        inplace=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        out = apply_rotary_pytorch(
            x,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            interleaved=interleaved,
            inplace=inplace,
        )
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(
                cos, sin, cu_seqlens
            )  # Can't save int with save_for_backward
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cu_seqlens, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        ctx.max_seqlen = max_seqlen
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cu_seqlens, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cu_seqlens = ctx.saved_tensors
        # TD [2023-09-02]: For some reason Triton (2.0.0.post1) errors with
        # "[CUDA]: invalid device context", and cloning makes it work. Idk why. Triton 2.1.0 works.
        if not ctx.interleaved and not ctx.inplace:
            do = do.clone()
        dx = apply_rotary_pytorch(
            do,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            interleaved=ctx.interleaved,
            inplace=ctx.inplace,
            conjugate=True,
        )
        return dx, None, None, None, None, None, None, None


def apply_rotary_emb(
    x,
    cos,
    sin,
    interleaved=False,
    inplace=False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
):
    """
    Arguments:
        x: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
        cos, sin: (seqlen_rotary, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        inplace: if True, apply rotary embedding in-place.
        seqlen_offsets: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Return:
        out: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding to the first rotary_dim of x.
    """
    return ApplyRotaryEmb.apply(
        x, cos, sin, interleaved, inplace, seqlen_offsets, cu_seqlens, max_seqlen
    )
