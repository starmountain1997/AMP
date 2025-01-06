from loguru import logger

from ..module_patcher import when_imported
from .llama import llama_rms_norm_forward, npu_apply_rotary_pos_emb


@when_imported("transformers")
def patch_qwen2(mod):
    logger.info("patched qwen2")
    mod.models.qwen2.modeling_qwen2.Qwen2RMSNorm.forward = llama_rms_norm_forward
    mod.models.qwen2.modeling_qwen2.apply_rotary_pos_emb = npu_apply_rotary_pos_emb

@when_imported("transformers")
def patch_qwen2_moe(mod):
    logger.info("patched qwen2_moe")
    mod.models.qwen2_moe.modeling_qwen2_moe.Qwen2RMSNorm.forward = llama_rms_norm_forward
    mod.models.qwen2_moe.modeling_qwen2_moe.apply_rotary_pos_emb = npu_apply_rotary_pos_emb 