import importlib
# https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/rotary.py
import importlib.machinery
import importlib.metadata
import sys
import types
from typing import Optional, Union

import torch


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
        y: (batch, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
    """
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
        device = x.device
    else:
        assert (
            max_seqlen is not None
        ), "If cu_seqlens is provided, max_seqlen must be given"
        total_seqlen, nheads, headdim = x.shape
        batch_p1 = cu_seqlens.size(0)
        batch = batch_p1 - 1
        seqlen = max_seqlen
        device = x.device

    seqlen_ro, d = cos.shape
    rotary_dim = 2 * d
    assert rotary_dim <= headdim, "rotary_dim must be <= headdim"
    assert sin.shape == (seqlen_ro, d), "sin must have the same shape as cos"
    assert x.dtype == cos.dtype == sin.dtype, "Input, cos, sin must have the same dtype"

    output = x if inplace else torch.empty_like(x)
    if not inplace and rotary_dim < headdim:
        output[..., rotary_dim:] = x[..., rotary_dim:]

    if is_varlen:
        for b in range(batch):
            # Get the start and end indices for the current batch
            start_idx = cu_seqlens[b]
            end_idx = cu_seqlens[b + 1]
            seqlen_b = end_idx - start_idx
            if seqlen_b == 0:
                continue  # Skip empty sequences

            # Extract the sequence for the current batch
            x_b = x[start_idx:end_idx]  # (seqlen_b, nheads, headdim)
            output_b = output[start_idx:end_idx]

            # Determine the offset for the current batch
            if isinstance(seqlen_offsets, torch.Tensor):
                offset_b = seqlen_offsets[b]
            else:
                offset_b = seqlen_offsets

            # Calculate positions and validate them
            pos = torch.arange(seqlen_b, device=device) + offset_b
            valid_pos = pos < seqlen_ro
            pos_clamped = pos.clamp(max=seqlen_ro - 1)

            # Get cos and sin values, applying validity mask
            cos_b = cos[pos_clamped].to(device)  # (seqlen_b, d)
            cos_b = torch.where(valid_pos.unsqueeze(-1), cos_b, 1.0)
            sin_b = sin[pos_clamped].to(device)
            sin_b = torch.where(valid_pos.unsqueeze(-1), sin_b, 0.0)

            if conjugate:
                sin_b = -sin_b

            # Expand dimensions for broadcasting over heads
            cos_b = cos_b.unsqueeze(1)  # (seqlen_b, 1, d)
            sin_b = sin_b.unsqueeze(1)

            # Apply rotary transformation
            x_rot = x_b[..., :rotary_dim]
            if interleaved:
                x_rot_pair = x_rot.view(seqlen_b, nheads, d, 2)
                x0, x1 = x_rot_pair.unbind(-1)
                o0 = x0 * cos_b - x1 * sin_b
                o1 = x0 * sin_b + x1 * cos_b
                x_rotated = torch.stack([o0, o1], dim=-1).view(
                    seqlen_b, nheads, rotary_dim
                )
            else:
                x0 = x_rot[..., :d]
                x1 = x_rot[..., d:]
                o0 = x0 * cos_b - x1 * sin_b
                o1 = x0 * sin_b + x1 * cos_b
                x_rotated = torch.cat([o0, o1], dim=-1)

            output_b[..., :rotary_dim] = x_rotated
    else:
        # Calculate positions for all batches and sequence elements
        if isinstance(seqlen_offsets, torch.Tensor):
            assert (
                seqlen_offsets.size(0) == batch
            ), "seqlen_offsets must have size (batch,)"
            offsets = seqlen_offsets.view(batch, 1)
        else:
            offsets = torch.tensor(seqlen_offsets, device=device).view(1, 1)

        pos = torch.arange(seqlen, device=device).view(1, seqlen) + offsets
        valid_pos = pos < seqlen_ro
        pos_clamped = pos.clamp(max=seqlen_ro - 1)

        # Gather cos and sin values, applying validity mask
        cos_pos = cos[pos_clamped].to(device)  # (batch, seqlen, d)
        cos_pos = torch.where(valid_pos.unsqueeze(-1), cos_pos, 1.0)
        sin_pos = sin[pos_clamped].to(device)
        sin_pos = torch.where(valid_pos.unsqueeze(-1), sin_pos, 0.0)

        if conjugate:
            sin_pos = -sin_pos

        # Expand dimensions for broadcasting over heads
        cos_pos = cos_pos.unsqueeze(2)  # (batch, seqlen, 1, d)
        sin_pos = sin_pos.unsqueeze(2)

        # Apply rotary transformation
        x_rot = x[..., :rotary_dim]
        if interleaved:
            x_rot_pair = x_rot.view(batch, seqlen, nheads, d, 2)
            x0, x1 = x_rot_pair.unbind(-1)
            o0 = x0 * cos_pos - x1 * sin_pos
            o1 = x0 * sin_pos + x1 * cos_pos
            x_rotated = torch.stack([o0, o1], dim=-1).view(
                batch, seqlen, nheads, rotary_dim
            )
        else:
            x0 = x_rot[..., :d]
            x1 = x_rot[..., d:]
            o0 = x0 * cos_pos - x1 * sin_pos
            o1 = x0 * sin_pos + x1 * cos_pos
            x_rotated = torch.cat([o0, o1], dim=-1)

        output[..., :rotary_dim] = x_rotated

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
