import importlib
import math
import os
from pathlib import Path
from typing import Type, Union

import torch
import torch.nn.functional as F
import torch_npu
from loguru import logger
from transformers.utils import HF_MODULES_CACHE

from ..module_patcher import when_imported


@torch.jit.script
def addmm_apply_rotary_pos_emb(x: torch.Tensor,
                               rope_cache: torch.Tensor) -> torch.Tensor:
    # TODO: 使用addmm替换
    # x: [b, np, sq, hn]
    b, np, sq, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:, :sq]
    xshaped = x.reshape(b, np, sq, rot_dim // 2, 2)
    rope_cache = rope_cache.view(-1, 1, sq, xshaped.size(3), 2)
    x_out2 = torch.stack([xshaped[..., 0] *
                          rope_cache[..., 0] -
                          xshaped[..., 1] *
                          rope_cache[..., 1], xshaped[..., 1] *
                          rope_cache[..., 0] +
                          xshaped[..., 0] *
                          rope_cache[..., 1], ], -
                         1, )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)


class NPURMSNorm(torch.nn.Module):
    def __init__(
            self,
            normalized_shape,
            eps=1e-5,
            device=None,
            dtype=None,
            **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(
                normalized_shape,
                device=device,
                dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        normalized_hidden_states, _ = torch_npu.npu_rms_norm(
            hidden_states, self.weight, epsilon=self.eps)
        return normalized_hidden_states.to(input_dtype)


class NPUCoreAttention(torch.nn.Module):
    def __init__(self, config, layer_number):
        super().__init__()
        self.config = config
        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.is_causal = True

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.coeff = coeff

        self.attention_dropout = torch.nn.Dropout(config.attention_dropout)

    def forward(
            self,
            query_layer,
            key_layer,
            value_layer,
            attention_mask):
        output_size = (
            query_layer.size(0),
            query_layer.size(1),
            query_layer.size(2),
            key_layer.size(2))
        query_layer = query_layer.view(
            output_size[0] * output_size[1], output_size[2], -1)
        matmul_input_buffer = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=query_layer.device)
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer,  # [b * np, sq, hn]
            torch_npu.npu_confusion_transpose(
                key_layer, (1, 2), (output_size[0] * output_size[1], output_size[3], -1), transpose_first=False),
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )
        attention_scores = matmul_result.view(*output_size)
        if self.attention_softmax_in_fp32:
            attention_scores = attention_scores.float()
        if self.coeff is not None:
            attention_scores = attention_scores * self.coeff
        if attention_mask is None and attention_scores.shape[2] == attention_scores.shape[3]:
            attention_mask = torch.ones(
                output_size[0],
                1,
                output_size[2],
                output_size[3],
                device=attention_scores.device,
                dtype=torch.bool)
            attention_mask.tril_()
        attention_mask = ~attention_mask
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                attention_mask, float("-inf"))
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = attention_probs.type_as(value_layer)
        attention_probs = self.attention_dropout(attention_probs)
        output_size = (
            value_layer.size(0),
            value_layer.size(1),
            query_layer.size(1),
            value_layer.size(3))
        value_layer = value_layer.view(
            output_size[0] * output_size[1], value_layer.size(2), -1)
        attention_probs = attention_probs.view(
            output_size[0] * output_size[1], output_size[2], -1)
        context_layer = torch.bmm(attention_probs, value_layer)
        context_layer = context_layer.view(*output_size)
        # [b, np, sq, hn] --> [b, sq, np, hn]
        context_layer = context_layer.transpose(1, 2).contiguous()
        new_context_layer_shape = context_layer.size(
        )[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.reshape(*new_context_layer_shape)
        return context_layer


def _config_to_kwargs(args):
    common_kwargs = {
        "dtype": args.torch_dtype,
    }
    return common_kwargs


class NPUMLP(torch.nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config, device=None):
        super().__init__()

        self.add_bias = config.add_bias_linear

        # Project to 4h. If using swiglu double the output width, see
        # https://arxiv.org/pdf/2002.05202.pdf
        self.dense_h_to_4h = torch.nn.Linear(
            config.hidden_size,
            config.ffn_hidden_size * 2,
            bias=self.add_bias,
            device=device,
            **_config_to_kwargs(config)
        )

        def swiglu(x):
            return torch_npu.npu_swiglu(x, dim=-1)

        self.activation_func = swiglu

        # Project back to h.
        self.dense_4h_to_h = torch.nn.Linear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias=self.add_bias,
            device=device,
            **_config_to_kwargs(config)
        )

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        return output


def patch_get_class_in_module(
        class_name: str, module_path: Union[str, os.PathLike]) -> Type:
    """
    Import a module on the cache directory for modules and extract a class from it.

    Args:
        class_name (`str`): The name of the class to import.
        module_path (`str` or `os.PathLike`): The path to the module to import.

    Returns:
        `typing.Type`: The class looked for.
    """
    name = os.path.normpath(module_path).replace(
        ".py", "").replace(
        os.path.sep, ".")
    module_path = str(Path(HF_MODULES_CACHE) / module_path)
    module = importlib.machinery.SourceFileLoader(
        name, module_path).load_module()
    package_name = name.split(".")[-1]
    if package_name == "modeling_chatglm":
        logger.info(f"{module} is patched.")
        # TODO: partial __init__
        module.RMSNorm = NPURMSNorm
        module.CoreAttention = NPUCoreAttention
        module.apply_rotary_pos_emb = addmm_apply_rotary_pos_emb
        module.MLP = NPUMLP
    return getattr(module, class_name)


@when_imported("transformers")
def patch_chatglm(mod):
    mod.dynamic_module_utils.get_class_in_module = patch_get_class_in_module
