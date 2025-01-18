import importlib
import math
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch_npu
from loguru import logger
from torch.nn import functional as F

from ..common.patch_transformers import patch_get_class_in_module
from ..module_patcher import when_imported
from .common.dummy_flash_attn import create_dummy_flash_attn
from .qwen2_vl import vision_flash_attention2_forward

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


def baichuan_whisper_attention_forward(
    self, hidden_states: torch.Tensor, seq_len: torch.Tensor
):
    bsz, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(bsz, self.num_heads, self.head_dim)
    key_states = self.k_proj(hidden_states).view(bsz, self.num_heads, self.head_dim)
    value_states = self.v_proj(hidden_states).view(bsz, self.num_heads, self.head_dim)

    cu_len = F.pad(torch.cumsum(seq_len, dim=0), (1, 0), "constant", 0).to(torch.int32)
    torch.max(seq_len).to(torch.int32).detach()

    head_num = query_states.shape[1]
    attn_output = torch_npu.npu_fusion_attention(
        query_states,
        key_states,
        value_states,
        head_num,
        pse=None,
        atten_mask=None,
        scale=1.0 / math.sqrt(query_states.shape[-1]),
        keep_prob=1,
        input_layout="TND",
        actual_seq_qlen=tuple(cu_len[1:].cpu().numpy().tolist()),
        actual_seq_kvlen=tuple(cu_len[1:].cpu().numpy().tolist()),
    )[0]
    attn_output = attn_output.reshape(bsz, self.embed_dim)
    attn_output = self.out_proj(attn_output)
    return attn_output


def baichuan_visual_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    causal_attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""

    bsz, tgt_len, embed_dim = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(
        bsz * tgt_len, self.num_heads, self.head_dim
    )
    key_states = self.k_proj(hidden_states).view(
        bsz * tgt_len, self.num_heads, self.head_dim
    )
    value_states = self.v_proj(hidden_states).view(
        bsz * tgt_len, self.num_heads, self.head_dim
    )

    # 暂时不考虑变长patch nums 固定长度为256/1024
    cu_len = torch.arange(
        0,
        (bsz + 1) * tgt_len,
        step=tgt_len,
        dtype=torch.int32,
        device=query_states.device,
    )
    # print(self.config.s2a, self.config.rope_scaling, cu_len, torch.sum(cu_len), q_len, kv_seq_len)
    # 如果不是f16 bf16不用flash attn
    if query_states.dtype in [torch.float16, torch.bfloat16]:
        head_num = query_states.shape[1]
        attn_output = torch_npu.npu_fusion_attention(
            query_states,
            key_states,
            value_states,
            head_num,
            pse=None,
            atten_mask=attention_mask,
            scale=1.0 / math.sqrt(query_states.shape[-1]),
            keep_prob=1,
            input_layout="TND",
            actual_seq_qlen=tuple(cu_len[1:].cpu().numpy().tolist()),
            actual_seq_kvlen=tuple(cu_len[1:].cpu().numpy().tolist()),
        )
        attn_output = attn_output.view(bsz, tgt_len, self.num_heads, self.head_dim)
    else:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False
        ):
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attention_mask, 0.0
            )
            attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output = self.out_proj(attn_output)

    return attn_output, None


@torch.no_grad()
def baichuan_visual_encoder_fake_input(self, device):
    merge_size = max(self.merge_size, self.config.spatial_merge_size)
    flatten_patches = torch.zeros(
        (
            merge_size * merge_size,
            3 * self.config.temporal_patch_size * self.config.patch_size**2,
        ),
        dtype=torch.float32,
        device=device,
    )
    return [flatten_patches], [(1, merge_size, merge_size)], [1]


def local_attention_forward(self, q, k, v, *args, use_flash=True, **kwargs):
    # input q,k,v [batch_size, num_head, seq_len, head_dim]
    # output [batch_size, seq_len, num_head, head_dim]
    if use_flash:
        q_len, num_heads = q.shape[2], q.shape[1]
        q = q.transpose(1, 2).reshape(-1, num_heads, self.head_dim)
        k = k.transpose(1, 2).reshape(-1, num_heads, self.head_dim)
        v = v.transpose(1, 2).reshape(-1, num_heads, self.head_dim)
        return q
        # return flash_attn_varlen_func(q,k,v,*args, **kwargs).reshape(-1,q_len, num_heads, self.head_dim)
    else:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False
        ):
            attn_output = F.scaled_dot_product_attention(q, k, v, *args, **kwargs)
        attn_output = attn_output.transpose(1, 2)
        return attn_output


def _patch_bc5_14b_omini(mod):
    package_name = mod.__name__.split(".")[-1]

    if package_name == "modeling_baichuan":
        logger.info(f"{mod} is patched.")
        mod.flash_attention_forward = flash_attention_forward_npu

        package_split = mod.__name__.split(".")
        package_split[-1] = "audio_modeling_baichuan"
        audio_modeling_baichuan_mod = ".".join(package_split)
        audio_modeling_baichuan_mod = importlib.import_module(
            audio_modeling_baichuan_mod
        )
        logger.info(f"{audio_modeling_baichuan_mod} is patched.")
        audio_modeling_baichuan_mod.BaichuanWhisperAttention.forward = (
            baichuan_whisper_attention_forward
        )

        package_split[-1] = "visual_modeling_baichuan"
        visual_modeling_baichuan_mod = ".".join(package_split)
        visual_modeling_baichuan_mod = importlib.import_module(
            visual_modeling_baichuan_mod
        )
        logger.info(f"{visual_modeling_baichuan_mod} is patched.")
        visual_modeling_baichuan_mod.BaichuanVisualAttention.forward = (
            baichuan_visual_attention_forward
        )
        visual_modeling_baichuan_mod.BaichuanVisualEncoder.fake_input = (
            baichuan_visual_encoder_fake_input
        )

        package_split[-1] = "sequence_parallel_utils"
        sequence_parallel_utils_mod = ".".join(package_split)
        sequence_parallel_utils_mod = importlib.import_module(
            sequence_parallel_utils_mod
        )
        logger.info(f"{sequence_parallel_utils_mod} is patched.")
        # sequence_parallel_utils_mod.LocalAttention.forward = local_attention_forward


@when_imported("transformers")
def patch_bc5_7b_omini(mod):
    os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"
    mod.modeling_flash_attention_utils._flash_supports_window_size = None
    mod.modeling_flash_attention_utils._upad_input = None
    mod.modeling_flash_attention_utils.prepare_fa2_from_position_ids = None
    mod.utils.is_flash_attn_2_available = lambda: True
    mod.utils.is_flash_attn_greater_or_equal_2_10 = lambda: False
    mod.models.qwen2_vl.modeling_qwen2_vl.VisionFlashAttention2.forward = (
        vision_flash_attention2_forward  
    )

    get_class_in_module_patched = patch_get_class_in_module(func=_patch_bc5_14b_omini)
    mod.dynamic_module_utils.get_class_in_module = get_class_in_module_patched

    create_dummy_flash_attn()
