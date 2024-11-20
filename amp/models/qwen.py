import torch_npu
from transformers.models.qwen.modeling_qwen2 import QWen2RMSNorm

from amp.utils import when_imported


class NPUQWen2RMSNorm(QWen2RMSNorm):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__(hidden_size, eps)

    def forward(self, hidden_states):
        return torch_npu.npu_rms_norm(
            hidden_states,
            self.weight,
            epsilon=self.variance_epsilon)[0]


@when_imported("transformers")
def patch_qwen(mod):
    mod.models.qwen.modeling_qwen2.QWen2RMSNorm = NPUQWen2RMSNorm
