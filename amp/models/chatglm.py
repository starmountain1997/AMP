from functools import partial

import torch
import torch.nn.functional as F
import torch_npu
import torch_npu.npu
from loguru import logger

from ..common.transformers import patch_get_class_in_module
from ..module_patcher import when_imported


@torch.jit.script
def addmm_apply_rotary_pos_emb(
    x: torch.Tensor, rope_cache: torch.Tensor
) -> torch.Tensor:
    # x: [b, np, sq, hn]
    b, np, sq, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]

    # Truncate to support variable sizes
    rope_cache = rope_cache[:, :sq]

    # Reshape x and rope_cache
    xshaped = x.reshape(b * np * sq, rot_dim // 2, 2)  # [b*np*sq, rot_dim//2, 2]
    rope_cache = rope_cache.view(1, sq, rot_dim // 2, 2).expand(
        b * np, -1, -1, -1
    )  # Broadcast for batch
    rope_cache = rope_cache.reshape(b * np * sq, rot_dim // 2, 2)

    # Apply rotation using matrix multiplication
    cos_theta, sin_theta = rope_cache[..., 0], rope_cache[..., 1]
    x_real, x_imag = xshaped[..., 0], xshaped[..., 1]
    x_out_real = x_real * cos_theta - x_imag * sin_theta  # Real part
    x_out_imag = x_imag * cos_theta + x_real * sin_theta  # Imaginary part

    # Combine results and reshape back
    # [b*np*sq, rot_dim//2, 2]
    x_out = torch.cat((x_out_real.unsqueeze(-1), x_out_imag.unsqueeze(-1)), dim=-1)
    x_out = x_out.flatten(2).reshape(b, np, sq, rot_dim)  # [b, np, sq, rot_dim]

    # Concatenate with x_pass
    return torch.cat((x_out, x_pass), dim=-1)


def patch_rms_norm_forward(self, hidden_states: torch.Tensor):
    input_dtype = hidden_states.dtype
    normalized_hidden_states, _ = torch_npu.npu_rms_norm(
        hidden_states, self.weight, epsilon=self.eps
    )
    return normalized_hidden_states.to(input_dtype)


def patch_core_attention_forward(
    self, query_layer, key_layer, value_layer, attention_mask
):
    # [b, np, sq, sk]
    output_size = (
        query_layer.size(0),
        query_layer.size(1),
        query_layer.size(2),
        key_layer.size(2),
    )
    # [b, np, sq, hn] -> [b * np, sq, hn]
    query_layer = query_layer.view(output_size[0] * output_size[1], output_size[2], -1)
    # [b, np, sk, hn] -> [b * np, sk, hn]
    key_layer = torch_npu.npu_confusion_transpose(
        key_layer,
        (0, 2, 1),
        (output_size[0] * output_size[1], output_size[3], key_layer.size(3)),
        # FIXME:
        transpose_first=False,
    )
    # key_layer = key_layer.view(output_size[0] * output_size[1], output_size[3], -1).transpose(1, 2)

    # preallocting input tensor: [b * np, sq, sk]
    matmul_input_buffer = torch.empty(
        output_size[0] * output_size[1],
        output_size[2],
        output_size[3],
        dtype=query_layer.dtype,
        device=query_layer.device,
    )

    # Raw attention scores. [b * np, sq, sk]
    matmul_result = torch.baddbmm(
        matmul_input_buffer,
        query_layer,  # [b * np, sq, hn]
        key_layer,  # [b * np, hn, sk]
        beta=0.0,
        alpha=(1.0 / self.norm_factor),
    )

    # change view to [b, np, sq, sk]
    attention_scores = matmul_result.view(*output_size)

    # ===========================
    # Attention probs and dropout
    # ===========================

    # attention scores and attention mask [b, np, sq, sk]
    if self.attention_softmax_in_fp32:
        attention_scores = attention_scores.float()
    if self.coeff is not None:
        attention_scores = attention_scores * self.coeff
    if (
        attention_mask is None
        and attention_scores.shape[2] == attention_scores.shape[3]
    ):
        attention_mask = torch.ones(
            output_size[0],
            1,
            output_size[2],
            output_size[3],
            device=attention_scores.device,
            dtype=torch.bool,
        )
        attention_mask.tril_()
        attention_mask = ~attention_mask
    if attention_mask is not None:
        attention_scores = attention_scores.masked_fill(attention_mask, float("-inf"))
    attention_probs = F.softmax(attention_scores, dim=-1)
    attention_probs = attention_probs.type_as(value_layer)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self.attention_dropout(attention_probs)

    # query layer shape: [b * np, sq, hn]
    # value layer shape: [b, np, sk, hn]
    # attention shape: [b, np, sq, sk]
    # context layer shape: [b, np, sq, hn]
    output_size = (
        value_layer.size(0),
        value_layer.size(1),
        query_layer.size(1),
        value_layer.size(3),
    )
    # change view [b * np, sk, hn]
    value_layer = value_layer.view(
        output_size[0] * output_size[1], value_layer.size(2), -1
    )
    # change view [b * np, sq, sk]
    attention_probs = attention_probs.view(
        output_size[0] * output_size[1], output_size[2], -1
    )
    # matmul: [b * np, sq, hn]
    context_layer = torch.bmm(attention_probs, value_layer)
    # change view [b, np, sq, hn]
    # [b, np, sq, hn] --> [b, sq, np, hn]
    context_layer = torch_npu.npu_confusion_transpose(
        context_layer, (0, 2, 1, 3), output_size, transpose_first=False
    ).contiguous()
    # [b, sq, np, hn] --> [b, sq, hp]
    new_context_layer_shape = context_layer.size()[:-2] + (
        self.hidden_size_per_partition,
    )
    context_layer = context_layer.reshape(*new_context_layer_shape)

    return context_layer


def _patch_chatglm(mod, name):
    package_name = name.split(".")[-1]
    if package_name == "modeling_chatglm":
        logger.info(f"{mod} is patched.")
        mod.RMSNorm.forward = patch_rms_norm_forward
        mod.CoreAttention.forward = patch_core_attention_forward
        mod.apply_rotary_pos_emb = addmm_apply_rotary_pos_emb
        mod.MLP.activation_func = lambda x: torch_npu.npu_swiglu(x, dim=-1)


@when_imported("transformers")
def patch_chatglm(mod):
    if not "4.39.0"<=mod.__version__ >= "4.40.2":
        raise ImportError(
                f"The version of transformers is {mod.__version__}, which is not supported. Please install version 4.39.0<=transformers<=4.40.2."
        )
    mod.dynamic_module_utils.get_class_in_module = partial(
        patch_get_class_in_module, func=_patch_chatglm
    )
