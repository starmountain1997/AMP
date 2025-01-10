from typing import Optional, Union

import torch

# https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/rotary.py
# https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/ptmoddevg/trainingmigrguide/performance_tuning_0023.html


def apply_rotary_positional_embeddings(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cu_seqlens: torch.Tensor = None,
    seqlen_offsets: torch.Tensor = None,
    is_varlen: bool = False,
    is_seqlen_offsets_tensor: bool = False,
    interleaved: bool = False,
    conjugate: bool = False,
) -> torch.Tensor:
    """
    Apply rotary positional embeddings to the input tensor x.

    Args:
        x (torch.Tensor): Input tensor of shape (batch, nheads, seqlen, headdim).
        cos (torch.Tensor): Precomputed cosine tensor of shape (batch, seqlen_ro, rotary_dim_half).
        sin (torch.Tensor): Precomputed sine tensor of shape (batch, seqlen_ro, rotary_dim_half).
        cu_seqlens (torch.Tensor, optional): Cumulative sequence lengths for variable-length sequences.
        seqlen_offsets (torch.Tensor or int, optional): Offsets for sequence lengths.
        is_varlen (bool, optional): Indicates if sequences have variable lengths.
        is_seqlen_offsets_tensor (bool, optional): Indicates if seqlen_offsets is a tensor.
        interleaved (bool, optional): Indicates if the data layout is interleaved.
        conjugate (bool, optional): Indicates if sine values should be conjugated (negated).

    Returns:
        torch.Tensor: Output tensor with rotary positional embeddings applied, same shape as x.
    """
    batch, nheads, seqlen, headdim = x.shape
    rotary_dim = (
        cos.shape[-1] * 2
    )  # since cos and sin are for half the rotary dimension
    rotary_dim_half = rotary_dim // 2

    # Ensure rotary_dim matches the headdim
    assert rotary_dim <= headdim, "Rotary dimension must be <= head dimension"

    # Prepare the output tensor
    out = x.clone()

    # Apply rotary embeddings
    if not is_varlen:
        # Fixed-length sequences
        # Expand cos and sin to match batch and nheads
        cos_expanded = cos.unsqueeze(1).expand(
            -1, nheads, -1, -1
        )  # (batch, nheads, seqlen_ro, rotary_dim_half)
        sin_expanded = sin.unsqueeze(1).expand(
            -1, nheads, -1, -1
        )  # (batch, nheads, seqlen_ro, rotary_dim_half)

        if seqlen_offsets is not None:
            if is_seqlen_offsets_tensor:
                # Offset per batch
                cos_expanded = cos_expanded + seqlen_offsets.view(-1, 1, 1)
                sin_expanded = sin_expanded + seqlen_offsets.view(-1, 1, 1)
            else:
                # Global offset
                cos_expanded = cos_expanded + seqlen_offsets
                sin_expanded = sin_expanded + seqlen_offsets

        # Handle conjugation
        if conjugate:
            sin_expanded = -sin_expanded

        # Apply rotary transformation
        # Split headdim into rotary and non-rotary parts
        x_rotary = x[..., :rotary_dim]
        x_rest = x[..., rotary_dim:]

        # Split rotary dimensions into two halves
        x0, x1 = x_rotary.chunk(
            2, dim=-1
        )  # Each of shape (batch, nheads, seqlen, rotary_dim_half)

        # Perform the rotary operation
        out_rotary0 = x0 * cos_expanded - x1 * sin_expanded
        out_rotary1 = x0 * sin_expanded + x1 * cos_expanded

        # Concatenate the rotated parts
        out_rotary = torch.cat([out_rotary0, out_rotary1], dim=-1)

        # Assign back to output tensor
        out = torch.cat([out_rotary, x_rest], dim=-1)

    else:
        # Variable-length sequences
        # cu_seqlens should be a 1D tensor of cumulative lengths with shape (batch + 1,)
        assert (
            cu_seqlens is not None
        ), "cu_seqlens must be provided for variable-length sequences"

        # Initialize output
        out = torch.zeros_like(x)

        for b in range(batch):
            start_idx = cu_seqlens[b].item()
            end_idx = cu_seqlens[b + 1].item()
            current_seqlen = end_idx - start_idx

            # Handle if seqlen_offsets is a tensor
            if is_seqlen_offsets_tensor and seqlen_offsets is not None:
                offset = seqlen_offsets[b].item()
            elif seqlen_offsets is not None:
                offset = seqlen_offsets
            else:
                offset = 0

            # Get the current sequence slice
            x_b = x[
                b, :, start_idx:end_idx, :rotary_dim
            ]  # (nheads, current_seqlen, rotary_dim)
            x_rest_b = x[
                b, :, start_idx:end_idx, rotary_dim:
            ]  # (nheads, current_seqlen, headdim - rotary_dim)

            # Get corresponding cos and sin
            cos_b = cos[
                b, :current_seqlen, :rotary_dim_half
            ]  # (current_seqlen, rotary_dim_half)
            sin_b = sin[
                b, :current_seqlen, :rotary_dim_half
            ]  # (current_seqlen, rotary_dim_half)

            # Apply offset if necessary
            if seqlen_offsets is not None:
                cos_b = cos_b + offset
                sin_b = sin_b + offset

            if conjugate:
                sin_b = -sin_b

            # Expand cos and sin to match nheads
            cos_b = cos_b.unsqueeze(0).expand(
                nheads, -1, -1
            )  # (nheads, current_seqlen, rotary_dim_half)
            sin_b = sin_b.unsqueeze(0).expand(
                nheads, -1, -1
            )  # (nheads, current_seqlen, rotary_dim_half)

            # Split rotary dimensions
            x0_b, x1_b = x_b.chunk(
                2, dim=-1
            )  # Each of shape (nheads, current_seqlen, rotary_dim_half)

            # Perform rotary transformation
            out_rotary0_b = x0_b * cos_b - x1_b * sin_b
            out_rotary1_b = x0_b * sin_b + x1_b * cos_b

            # Concatenate the rotated parts
            out_rotary_b = torch.cat(
                [out_rotary0_b, out_rotary1_b], dim=-1
            )  # (nheads, current_seqlen, rotary_dim)

            # Concatenate back the rotary and non-rotary parts
            out_b = torch.cat(
                [out_rotary_b, x_rest_b], dim=-1
            )  # (nheads, current_seqlen, headdim)

            # Assign to output
            out[b, :, start_idx:end_idx, :] = out_b

    return out


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

    batch, nheads, seqlen, headdim = x.shape

    batch_ro, seqlen_ro, rotary_dim = cos.shape

    assert batch == batch_ro
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

    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])

    output = apply_rotary_positional_embeddings(
        x=x, cos=cos, sin=sin, is_varlen=False, conjugate=conjugate
    )
