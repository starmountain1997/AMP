import sys

import torch
import torch_npu
from loguru import logger

from ..common.patch_transformers import patch_get_class_in_module
from ..module_patcher import when_imported


class FastGelu(torch.nn.GELU):
    def forward(self, input_data):
        return torch_npu.fast_gelu(input_data)


def _patch_cogagent2_9b(mod):
    package_name = mod.__name__.split(".")[-1]
    if package_name == "modeling_chatglm":
        logger.info(f"{mod} is patched.")
        mod.MLP.activation_func = lambda x: torch_npu.npu_swiglu(x, dim=-1)
        from .chatglm import patch_rms_norm_forward

        mod.RMSNorm.forward = patch_rms_norm_forward

        parts = mod.__name__.split(".")
        parts[-1] = "visual"
        new_name = ".".join(parts)
        new_mod = sys.modules.get(new_name)
        logger.info(f"{new_mod} is patched.")
        new_mod.GLU.act1 = FastGelu()


@when_imported("transformers")
def patch_cogagent2_9b(mod):
    get_class_in_module_patched = patch_get_class_in_module(func=_patch_cogagent2_9b)
    mod.dynamic_module_utils.get_class_in_module = get_class_in_module_patched
