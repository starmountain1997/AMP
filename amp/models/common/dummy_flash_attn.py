import importlib
# https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/rotary.py
import importlib.machinery
import importlib.metadata
import sys
import types
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


def create_dummy_flash_attn():
    flash_attn = types.ModuleType("flash_attn")
    sys.modules["flash_attn"] = flash_attn
    flash_attn.__spec__ = importlib.machinery.ModuleSpec(
        name="flash_attn",
        loader=None,
        origin="<dynamic>",
    )
    sys.modules["flash_attn"].__version__ = "0.0.1"

    flash_attn.layers = types.ModuleType("flash_attn.layers")
    sys.modules["flash_attn.layers"] = flash_attn.layers
    flash_attn.layers.__spec__ = importlib.machinery.ModuleSpec(
        name="flash_attn.layers",
        loader=None,
        origin="<dynamic>",
    )

    flash_attn.layers.rotary = types.ModuleType("flash_attn.layers.rotary")
    sys.modules["flash_attn.layers.rotary"] = flash_attn.layers.rotary
    flash_attn.layers.rotary.__spec__ = importlib.machinery.ModuleSpec(
        name="flash_attn.layers.rotary",
        loader=None,
        origin="<dynamic>",
    )

    flash_attn.bert_padding = types.ModuleType("flash_attn.bert_padding")
    sys.modules["flash_attn.bert_padding"] = flash_attn.bert_padding
    flash_attn.bert_padding.__spec__ = importlib.machinery.ModuleSpec(
        name="flash_attn.bert_padding",
        loader=None,
        origin="<dynamic>",
    )

    # Assign functions or attributes to the submodules
    flash_attn.layers.rotary.apply_rotary_emb_func = apply_rotary_emb
    flash_attn.bert_padding.index_first_axis = None
    flash_attn.bert_padding.pad_input = None
    flash_attn.bert_padding.unpad_input = None
    flash_attn.flash_attn_func = None
    flash_attn.flash_attn_varlen_func = None
    flash_attn.flash_attn_with_kvcache = None

    decord = types.ModuleType("decord")
    sys.modules["decord"] = decord
    decord.__spec__ = importlib.machinery.ModuleSpec(
        name="decord",
        loader=None,
        origin="<dynamic>",
    )

    decord.cpu = None
    decord.VideoReader = None
