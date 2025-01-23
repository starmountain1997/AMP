from dummy_fa_openai import apply_rotary_pytorch
from flash_attn.ops.triton.rotary import apply_rotary
import torch

def test_apply_rotary_pytorch():
    batch = 2
    seqlen = 4
    nheads = 3
    headdim = 8
    rotary_dim = 4
    seqlen_ro = 4

    device="cuda"

    x = torch.rand(batch, seqlen, nheads, headdim, dtype=torch.float32,device=device)

    cos = torch.rand(seqlen_ro, rotary_dim // 2, dtype=torch.float32,device=device)
    sin = torch.rand(seqlen_ro, rotary_dim // 2, dtype=torch.float32,device=device)

    seqlen_offsets = 0  # Can also be a tensor of shape (batch,)
    cu_seqlens = None  # Can also be a tensor of shape (batch + 1,)
    max_seqlen = seqlen
    interleaved = False
    inplace = False
    conjugate = False

    # Call the function
    output_1 = apply_rotary(
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
    output_2 = apply_rotary_pytorch(
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

    print("\nOutput1:")
    print(output_1)
    
    print("\nOutput2:")
    print(output_2)
    

if __name__=="__main__":
    test_apply_rotary_pytorch()