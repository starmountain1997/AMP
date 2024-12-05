import torch_npu

from ..module_patcher import when_imported
from .llama import npu_apply_rotary_pos_emb


def patch_qwen2_rms_norm_forward(self, hidden_states):
    return torch_npu.npu_rms_norm(
        hidden_states, self.weight, epsilon=self.variance_epsilon
    )[0]


@when_imported("transformers")
def patch_qwen(mod):
    mod.models.qwen2.modeling_qwen2.Qwen2RMSNorm.forward = patch_qwen2_rms_norm_forward
    mod.models.qwen2.modeling_qwen2.apply_rotary_pos_emb = npu_apply_rotary_pos_emb
