import torch
from loguru import logger
from ..module_patcher import when_imported
from ..common.transformers import patch_get_class_in_module
from functools import partial
import sys

def apply_rotary_pos_emb_no_jit(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [sq, b, np, hn]
    sq, b, np, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:sq]
    xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
    rope_cache = rope_cache.view(sq, -1, 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)

def _patch_characterglm_6b(mod, name):
    package_name = name.split(".")[-1]
    if package_name == "modeling_characterglm":
        parts=name.split(".")
        parts[-1]="modeling_chatglm"
        new_name=".".join(parts)
        new_mod=sys.modules.get(new_name)
        logger.info(f"{new_mod} is patched.")

        new_mod.apply_rotary_pos_emb = apply_rotary_pos_emb_no_jit


@when_imported("transformers")
def patch_characterglm_6b(mod):
    mod.dynamic_module_utils.get_class_in_module = partial(
        patch_get_class_in_module, func=_patch_characterglm_6b
    )
