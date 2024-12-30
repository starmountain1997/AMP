from loguru import logger

from ..module_patcher import when_imported
from .llama import llama_rms_norm_forward, npu_apply_rotary_pos_emb


@when_imported("transformers")
def patch_deepseek_coder(mod):
    """
    已在 deepseek-coder-6.7b-instruct       模型上测试通过
         deepseek-coder-33b-instruct
         deepseek-coder-6.7b-base
         deepseek-coder-7b-instruct-v1.5
         
    """
    logger.info("patched deepseek_coder")
    mod.models.llama.modeling_llama.LlamaRMSNorm.forward = llama_rms_norm_forward
    mod.models.llama.modeling_llama.apply_rotary_pos_emb = npu_apply_rotary_pos_emb
