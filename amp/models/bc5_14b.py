import math
import os
import sys
import types
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


def _patch_bc5_14b(mod):
    package_name = mod.__name__.split(".")[-1]
    # Create a dummy `xformers` module
    flash_attn = types.ModuleType("flash_attn")
    flash_attn.layers = types.ModuleType("flash_attn.layers")
    flash_attn.layers.rotary = types.ModuleType("flash_attn.layers.rotary")
    flash_attn.layers.rotary.apply_rotary_emb_func = apply_rotary_emb
    sys.modules["flash_attn"] = flash_attn
    sys.modules["flash_attn.layers"] = flash_attn.layers
    sys.modules["flash_attn.layers.rotary"] = flash_attn.layers.rotary

    if package_name == "modeling_baichuan":
        logger.info(f"{mod} is patched.")
        mod.flash_attention_forward = flash_attention_forward_npu


@when_imported("transformers")
def patch_bc5_14b(mod):
    os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"
    get_class_in_module_patched = patch_get_class_in_module(func=_patch_bc5_14b)
    mod.dynamic_module_utils.get_class_in_module = get_class_in_module_patched
