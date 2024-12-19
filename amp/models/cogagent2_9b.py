import torch

import importlib
import sys
import types

import torch
import torch.nn.functional as F
from loguru import logger

from ..common.patch_transformers import patch_get_class_in_module
from ..module_patcher import when_imported
import torch_npu

def patch_apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [b, np, sq, hn]
    sq =  x.size(1)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:, :sq]
    xshaped = x.chunk(2, -1)
    cos, sin = rope_cache[...,0].unsqueeze(2), rope_cache[...,1].unsqueeze(2)
    x_out2 = torch.concat(
        [
            xshaped[0] * cos - xshaped[1] * sin,
            xshaped[1] * cos + xshaped[0] * sin,
        ],
        -1,
    )
    return torch.cat((x_out2, x_pass), dim=-1)



class FastGelu(torch.nn.GELU):
    def forward(self, input_data):
        return torch_npu.fast_gelu(input_data)


def _patch_cogagent2_9b(mod):
    package_name = mod.__name__.split(".")[-1]
    if package_name == "modeling_chatglm":
        logger.info(f"{mod} is patched.")
        mod.MLP.activation_func = lambda x: torch_npu.npu_swiglu(x, dim=-1)
        from .chatglm import patch_rms_norm_forward, addmm_apply_rotary_pos_emb
        mod.RMSNorm.forward = patch_rms_norm_forward

        parts = mod.__name__.split(".")
        parts[-1] = "visual"
        new_name = ".".join(parts)
        new_mod = sys.modules.get(new_name)
        logger.info(f"{new_mod} is patched.")
        new_mod.GLU.act1=FastGelu()




@when_imported("transformers")
def patch_cogagent2_9b(mod):
    
    get_class_in_module_patched = patch_get_class_in_module(
        func=_patch_cogagent2_9b
    )
    mod.dynamic_module_utils.get_class_in_module = get_class_in_module_patched
