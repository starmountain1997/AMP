import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch_npu
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import repeat_kv
from transformers.utils import logging

from amp.module_patcher import when_imported

logger = logging.get_logger(__name__)


def llama_rms_norm_forward(self, hidden_states):
    return torch_npu.npu_rms_norm(
        hidden_states, self.weight, epsilon=self.variance_epsilon)[0]


def npu_apply_rotary_pos_emb(
        q,
        k,
        cos,
        sin,
        position_ids=None,
        unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    return q_embed, k_embed


def llama_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    # will become mandatory in v4.46
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads *
                             self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            F.linear(
                hidden_states,
                query_slices[i]) for i in range(
                self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            F.linear(
                hidden_states,
                key_slices[i]) for i in range(
                self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            F.linear(
                hidden_states,
                value_slices[i]) for i in range(
                self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = torch_npu.npu_confusion_transpose(
        query_states,
        (0, 2, 1, 3),
        (bsz,
         q_len,
         self.num_heads,
         self.head_dim),
        transpose_first=False)
    key_states = torch_npu.npu_confusion_transpose(
        key_states,
        (0, 2, 1, 3),
        (bsz,
         q_len,
         self.num_key_value_heads,
         self.head_dim),
        transpose_first=False)
    value_states = torch_npu.npu_confusion_transpose(
        value_states,
        (0, 2, 1, 3),
        (bsz,
         q_len,
         self.num_key_value_heads,
         self.head_dim),
        transpose_first=False)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory.")
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = npu_apply_rotary_pos_emb(
        query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed
        # for the static cache
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(
        2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(
        attn_weights,
        dim=-1,
        dtype=torch.float32).to(
        query_states.dtype)
    attn_weights = nn.functional.dropout(
        attn_weights,
        p=self.attention_dropout,
        training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}")

    attn_output = torch_npu.npu_confusion_transpose(
        attn_output,
        (0, 2, 1, 3),
        (bsz, q_len, self.num_heads * self.head_dim),
        transpose_first=True)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(
            self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(
            self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i])
                          for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


@when_imported("transformers")
def patch_llama(mod):
    logger.info("patched llama")
    mod.models.llama.modeling_llama.LlamaRMSNorm.forward = llama_rms_norm_forward
    mod.models.llama.modeling_llama.apply_rotary_pos_emb = npu_apply_rotary_pos_emb
    mod.models.llama.modeling_llama.LlamaAttention.forward = llama_attention_forward
