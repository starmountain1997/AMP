import sys
import types

from loguru import logger

from ..common.patch_transformers import patch_get_class_in_module
from ..module_patcher import when_imported
from .rotary_kernel import apply_rotary_pytorch


def dummy_rotary_kernel():
    pass


def jit(fn):
    return fn


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
        logger.warning("Patching CogVLM2-Llama3-Chat-19B model")
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


@when_imported("transformers")
def patch_cogvlm2_llama3_chat_19b(mod):
    get_class_in_module_patched = patch_get_class_in_module(
        func=_patch_cogvlm2_llama3_chat_19b
    )
    mod.dynamic_module_utils.get_class_in_module = get_class_in_module_patched
