import torch

from amp.models.common.dummy_flash_attn import apply_rotary_pytorch


def test_apply_rotary_pytorch():
    batch = 2
    seqlen = 4
    nheads = 3
    headdim = 8
    rotary_dim = 4
    seqlen_ro = 4

    device = "npu:0"

    x = torch.rand(batch, seqlen, nheads, headdim, dtype=torch.float32, device=device)

    cos = torch.rand(seqlen_ro, rotary_dim // 2, dtype=torch.float32, device=device)
    sin = torch.rand(seqlen_ro, rotary_dim // 2, dtype=torch.float32, device=device)

    seqlen_offsets = 0  # Can also be a tensor of shape (batch,)
    cu_seqlens = None  # Can also be a tensor of shape (batch + 1,)
    max_seqlen = seqlen
    interleaved = False
    inplace = False
    conjugate = False

    # Call the function
    output = apply_rotary_pytorch(
        x=x,
        cos=cos,
        sin=sin,
        seqlen_offsets=seqlen_offsets,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        interleaved=interleaved,
        inplace=inplace,
        conjugate=conjugate,
    )

    print("Input x:")
    print(x)
    print("\nCosine:")
    print(cos)
    print("\nSine:")
    print(sin)
    print("\nOutput:")
    print(output)


if __name__ == "__main__":
    test_apply_rotary_pytorch()
