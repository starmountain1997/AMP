from functools import partial
from typing import Optional

import torch.nn.functional as F
import torch_npu
import torch_npu.npu
from loguru import logger
from torch import Tensor

from ..common.patch_transformers import patch_get_class_in_module
from ..module_patcher import when_imported


def llama_rms_norm_forward(self, hidden_states):
    return torch_npu.npu_rms_norm(
        hidden_states, self.weight, epsilon=self.variance_epsilon
    )[0]


def multi_header_attention_qkv_attention(
    self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
):
    logger.info("patching multi_header_attention_qkv_attention")
    n_batch, n_ctx, n_state = q.shape
    scale = (n_state // self.n_head) ** -0.25
    q = (
        torch_npu.npu_confusion_transpose(
            q,
            (0, 2, 1, 3),
            (*q.shape[:2], self.n_head, q.shape[2] // self.n_head),
            transpose_first=False,
        )
        * scale
    )
    k = (
        torch_npu.npu_confusion_transpose(
            k,
            (0, 2, 3, 1),
            (*k.shape[:2], self.n_head, k.shape[2] // self.n_head),
            transpose_first=False,
        )
        * scale
    )
    v = torch_npu.npu_confusion_transpose(
        v,
        (0, 2, 1, 3),
        (*v.shape[:2], self.n_head, v.shape[2] // self.n_head),
        transpose_first=False,
    )

    qk = q @ k
    if mask is not None:
        qk += mask

    w = F.softmax(qk, dim=-1).to(q.dtype)
    return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


def _patch_mergrezo(mod, name):
    package_name = name.split(".")[-1]
    if package_name == "audio":
        logger.info(f"{mod} is patched.")
        mod.MultiHeadAttention.qkv_attention = multi_header_attention_qkv_attention


@when_imported("transformers")
def patch_mergrezo(mod):
    mod.models.llama.modeling_llama.LlamaRMSNorm.forward = llama_rms_norm_forward
    mod.dynamic_module_utils.get_class_in_module = partial(
        patch_get_class_in_module, func=_patch_mergrezo
    )
