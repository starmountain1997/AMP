from loguru import logger

from ..module_patcher import when_imported
from .llama import llama_rms_norm_forward, npu_apply_rotary_pos_emb

# https://modelers.cn/models/INFINIGENCE-AI/Megrez-3B-Omni


@when_imported("transformers")
def patch_megrezo_3b_omni(mod):
    logger.info("patched llama for megrezo-3b-omni")
    mod.models.llama.modeling_llama.LlamaRMSNorm.forward = llama_rms_norm_forward
    mod.models.llama.modeling_llama.apply_rotary_pos_emb = npu_apply_rotary_pos_emb
