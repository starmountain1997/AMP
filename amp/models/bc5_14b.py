import math
import os
from typing import Optional

import numpy as np
import torch
import torch_npu
from loguru import logger

from ..common.patch_transformers import patch_get_class_in_module
from ..module_patcher import when_imported
from .common.dummy_flash_attn import apply_rotary_emb
# https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/ptmoddevg/trainingmigrguide/performance_tuning_0027.html


def flash_attention_forward_npu(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    seqlens: Optional[torch.LongTensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: bool = None,
):
    device = query_states.device

    if not use_top_left_mask:
        causal = is_causal
    else:
        causal = is_causal and query_length != 1

    atten_mask = (
        torch.from_numpy(np.triu(np.ones([2048, 2048]), k=1)).bool().to(device)
        if causal
        else None
    )  # FIXME:
    sparse_mode = 3 if causal else 2
    scale = (
        softmax_scale
        if softmax_scale is not None
        else 1.0 / math.sqrt(query_states.shape[-1])
    )

    if seqlens is not None:
        raise NotImplementedError
    elif attention_mask is not None:
        raise NotImplementedError
    else:
        attn_output = torch_npu.npu_fusion_attention(
            query_states,
            key_states,
            value_states,
            head_num=query_states.shape[2],
            input_layout="BSND",
            pse=None,
            keep_prob=1.0,
            scale=scale,
            atten_mask=atten_mask,
            sparse_mode=sparse_mode,
        )[0]

    return attn_output


def rotary_embedding_forward(
    self, q, k, seqlen_offset=None, cu_seqlens=None, max_seqlen=None
):
    # x: [bs, num_attention_heads, seq_len, head_size]
    # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
    seq_len_dim = 1
    seq_len = q.shape[seq_len_dim] + seqlen_offset
    if seq_len > self.max_seq_len_cached:
        self.max_seq_len_cached = seq_len
        self.inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2).float().to(self.inv_freq.device) / self.dim
            )
        )
        t = torch.arange(
            self.max_seq_len_cached,
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype,
        )
        # freqs = torch.einsum("i,j->ij", t, self.inv_freq) # dont use this, bug in fp16
        freqs = torch.outer(t, self.inv_freq)
        self.cos_cached = freqs.cos().to(q.device)
        self.sin_cached = freqs.sin().to(k.device)
    q_ori_size = q.size()
    k_ori_size = k.size()
    if cu_seqlens is not None:
        q = flatten_one_dim(q)
        k = flatten_one_dim(k)
    q_new = apply_rotary_emb(
        q.float(),
        self.cos_cached[seqlen_offset:],
        self.sin_cached[seqlen_offset:],
        self.interleaved,
        True,  # inplace=True
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
    ).to(q.dtype)
    k_new = apply_rotary_emb(
        k.float(),
        self.cos_cached[seqlen_offset:],
        self.sin_cached[seqlen_offset:],
        self.interleaved,
        True,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
    ).to(k.dtype)
    if cu_seqlens is not None:
        q_new = q_new.reshape(*q_ori_size)
        k_new = k_new.reshape(*k_ori_size)
    return q_new, k_new


def _patch_bc5_14b(mod):
    package_name = mod.__name__.split(".")[-1]

    if package_name == "modeling_baichuan":
        logger.info(f"{mod} is patched.")
        mod.flash_attention_forward = flash_attention_forward_npu
        mod.RotaryEmbedding.forward = rotary_embedding_forward


@when_imported("transformers")
def patch_bc5_14b(mod):
    os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"
    get_class_in_module_patched = patch_get_class_in_module(func=_patch_bc5_14b)
    mod.dynamic_module_utils.get_class_in_module = get_class_in_module_patched
