import os
import sys
import types

import torch
import torch_npu
from loguru import logger

from ..common.patch_transformers import patch_get_class_in_module
from ..module_patcher import when_imported
from .common.dummy_flash_attn import apply_rotary_pytorch
from .common.index_input import patch_index_input


def dummy_rotary_kernel():
    pass


def jit(fn):
    return fn


class FastGelu(torch.nn.GELU):
    def forward(self, input_data):
        return torch_npu.fast_gelu(input_data)


def rms_norm_forward(self, hidden_states):
    input_dtype = hidden_states.dtype
    return torch_npu.npu_rms_norm(
        hidden_states,
        self.weight,  # match input's float type for weight
        epsilon=self.variance_epsilon,
    )[0].to(input_dtype)


def npu_rotate_half(x):
    r1 = torch.zeros_like(x)
    r2 = torch.ones_like(x)

    rotated_output = torch_npu.npu_rotary_mul(x, r1, r2)
    return rotated_output


def _patch_cogvlm2_llama3_chat_19b(mod):
    package_name = mod.__name__.split(".")[-1]
    # Create a dummy `triton` module
    triton = types.ModuleType("triton")
    triton.language = types.ModuleType("triton.language")
    triton.language.constexpr = types.ModuleType("triton.language.constexpr")

    triton.jit = jit
    sys.modules["triton"] = triton
    sys.modules["triton.language"] = triton.language
    sys.modules["triton.language.constexpr"] = triton.language.constexpr
    sys.modules["triton.jit"] = triton.jit

    # Create a dummy `xformers` module
    xformers = types.ModuleType("xformers")
    xformers.ops = types.ModuleType("xformers.ops")
    sys.modules["xformers"] = xformers
    sys.modules["xformers.ops"] = xformers.ops

    if package_name == "modeling_cogvlm":
        logger.info(f"{mod} is patched.")
        patch_index_input(mod, mod.CogVLMModel.forward)
        mod.RMSNorm.forward = rms_norm_forward  # 调优
        mod.rotate_half = npu_rotate_half  # 调优

        package_split = mod.__name__.split(".")
        package_split[-1] = "util"
        util_mod = ".".join(package_split)
        logger.warning(f"Patching {mod.__name__} with {util_mod}")
        import importlib

        util_mod = importlib.import_module(util_mod)
        util_mod.apply_rotary = apply_rotary_pytorch
        util_mod.rotary_kernel = dummy_rotary_kernel

        package_split = mod.__name__.split(".")
        package_split[-1] = "visual"
        visual_mod = ".".join(package_split)
        visual_mod = importlib.import_module(visual_mod)
        logger.info(f"{visual_mod} is patched.")
        from .cogvlm_chat_hf import attention_forward_visual

        visual_mod.Attention.forward = attention_forward_visual
        visual_mod.GLU.act1 = FastGelu()  # 调优


@when_imported("transformers")
def patch_cogvlm2_llama3_chat_19b(mod):
    os.environ["TASK_QUEUE_ENABLE"] = "2"
    os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"

    get_class_in_module_patched = patch_get_class_in_module(
        func=_patch_cogvlm2_llama3_chat_19b
    )
    mod.dynamic_module_utils.get_class_in_module = get_class_in_module_patched
