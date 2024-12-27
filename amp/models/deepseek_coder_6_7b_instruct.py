import torch_npu

from ..module_patcher import when_imported
from .llama import npu_apply_rotary_pos_emb, llama_rms_norm_forward

from loguru import logger



@when_imported("transformers")
def patch_deepseek_coder_6_7b_instruct(mod):
    logger.info("patched deepseek_coder_6_7b_instruct")
    mod.models.llama.modeling_llama.LlamaRMSNorm.forward = llama_rms_norm_forward
    mod.models.llama.modeling_llama.apply_rotary_pos_emb = npu_apply_rotary_pos_emb
