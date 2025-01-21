import math

import torch
import torch_npu
from transformers.models.qwen2_vl.modeling_qwen2_vl import \
    apply_rotary_pos_emb_vision


def vision_flash_attention2_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: torch.Tensor = None,
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    q, k, v = (
        self.qkv(hidden_states)
        .reshape(seq_length, 3, self.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
    k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    head_num = q.shape[1]
    attn_output = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        head_num,
        pse=None,
        atten_mask=None,
        scale=1.0 / math.sqrt(q.shape[-1]),
        keep_prob=1,
        input_layout="TND",
        actual_seq_qlen=tuple(cu_seqlens[1:].cpu().numpy().tolist()),
        actual_seq_kvlen=tuple(cu_seqlens[1:].cpu().numpy().tolist()),
    )[0].reshape(seq_length, -1)
    attn_output = self.proj(attn_output)
    return attn_output
