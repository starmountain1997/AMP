import torch_npu
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from ..module_patcher import when_imported
from .llama import npu_apply_rotary_pos_emb


class NPUQwen2RMSNorm(Qwen2RMSNorm):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__(hidden_size, eps)

    def forward(self, hidden_states):
        return torch_npu.npu_rms_norm(
            hidden_states,
            self.weight,
            epsilon=self.variance_epsilon)[0]


@when_imported("transformers")
def patch_qwen(mod):
    mod.models.qwen2.modeling_qwen2.Qwen2RMSNorm = NPUQwen2RMSNorm
    mod.models.qwen2.modeling_qwen2.apply_rotary_pos_emb = npu_apply_rotary_pos_emb
