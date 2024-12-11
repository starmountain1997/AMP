import sys

import torch
import torch_npu
from loguru import logger

from ..common.patch_transformers import patch_get_class_in_module
from ..module_patcher import when_imported


def rms_norm_forward(self, hidden_states: torch.Tensor):
    input_dtype = hidden_states.dtype
    output = torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.eps)[0]
    return output.to(input_dtype)


def _patch_characterglm_6b(mod, name):
    # https://modelers.cn/models/zhipuai/characterglm-6b/tree/main
    logger.debug(mod)
    logger.debug(name)
    package_name = name.split(".")[-1]
    if package_name == "modeling_characterglm":
        parts = name.split(".")
        parts[-1] = "modeling_chatglm"
        new_name = ".".join(parts)
        new_mod = sys.modules.get(new_name)
        logger.info(f"{new_mod} is patched.")
        new_mod.RMSNorm.forward = rms_norm_forward


@when_imported("transformers")
def patch_characterglm_6b(mod):
    if mod.__version__ != "4.41.2":
        raise ImportError(
            "when running characterglm_6b, please install transformers==4.41.2"
        )
    get_class_in_module_patched = patch_get_class_in_module(func=_patch_characterglm_6b)
    mod.dynamic_module_utils.get_class_in_module = get_class_in_module_patched
