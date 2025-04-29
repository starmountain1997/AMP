import importlib
import sys
import types

import torch
import torch.nn.functional as F
from loguru import logger

from ..common.patch_transformers import patch_get_class_in_module
from ..module_patcher import when_imported


def attention_forward_visual(self, x: "tensor(B, L, D)") -> "tensor(B, L, D)":
    # https://github.com/THUDM/CogVLM2/issues/56
    # https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/__init__.py#L194
    B, L, _ = x.shape
    qkv = self.query_key_value(x)
    qkv = qkv.reshape(B, L, 3, self.num_heads, -1).permute(
        2, 0, 1, 3, 4
    )  # 3, B, L, H, D
    q, k, v = qkv[0], qkv[1], qkv[2]

    scale = 1.0 / q.shape[-1] ** 0.5
    q = q * scale
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(-1)
    attn = F.dropout(attn)
    attn = attn @ v
    out = attn.transpose(1, 2)

    output = self.dense(out.contiguous().view(B, L, -1))
    output = self.output_dropout(output)
    return output


def attention_forward_cross_visual(self, x, rel_pos_bias=None, attn_mask=None):
    B, N, C = x.shape
    if self.subln:
        if self.q_proj.weight.dtype == torch.uint8:
            import bitsandbytes as bnb

            q = bnb.matmul_4bit(
                x,
                self.q_proj.weight.t(),
                bias=self.q_bias,
                quant_state=self.q_proj.weight.quant_state,
            )
            k = bnb.matmul_4bit(
                x,
                self.k_proj.weight.t(),
                bias=None,
                quant_state=self.k_proj.weight.quant_state,
            )
            v = bnb.matmul_4bit(
                x,
                self.v_proj.weight.t(),
                bias=self.v_bias,
                quant_state=self.v_proj.weight.quant_state,
            )
        else:
            q = F.linear(input=x, weight=self.q_proj.weight, bias=self.q_bias)
            k = F.linear(input=x, weight=self.k_proj.weight, bias=None)
            v = F.linear(input=x, weight=self.v_proj.weight, bias=self.v_bias)

        q = q.reshape(B, N, self.num_heads, -1).permute(
            0, 2, 1, 3
        )  # B, num_heads, N, C
        k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    else:
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )

        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(
            2, 0, 3, 1, 4
        )  # 3, B, num_heads, N, C
        q, k, v = qkv[0], qkv[1], qkv[2]

    if self.rope:
        # slightly fast impl
        q_t = q[:, :, 1:, :]
        ro_q_t = self.rope(q_t)
        q = torch.cat((q[:, :, :1, :], ro_q_t), -2).type_as(v)

        k_t = k[:, :, 1:, :]
        ro_k_t = self.rope(k_t)
        k = torch.cat((k[:, :, :1, :], ro_k_t), -2).type_as(v)

    if self.xattn:
        q = q.permute(0, 2, 1, 3)  # B, num_heads, N, C -> B, N, num_heads, C
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # scale = 1.0 / q.shape[-1] ** 0.5
        q = q * self.scale
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(-1)
        attn = F.dropout(attn, self.xattn_drop)
        attn = attn @ v
        x = attn.transpose(1, 2)

        x = x.reshape(B, N, -1)
        x = self.inner_attn_ln(x)
        x = self.proj(x)
        x = self.proj_drop(x)
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if self.relative_position_bias_table is not None:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1,
                -1,
            )  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1
            ).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0).type_as(attn)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias.type_as(attn)

        if attn_mask is not None:
            attn_mask = attn_mask.bool()
            attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.inner_attn_ln(x)
        x = self.proj(x)
        x = self.proj_drop(x)
    return x


def _patch_cogvlm_chat_hf(mod):
    package_name = mod.__name__.split(".")[-1]
    # Create a dummy `xformers` module
    xformers = types.ModuleType("xformers")
    xformers.ops = types.ModuleType("xformers.ops")
    sys.modules["xformers"] = xformers
    sys.modules["xformers.ops"] = xformers.ops

    if package_name == "modeling_cogvlm":
        package_split = mod.__name__.split(".")
        package_split[-1] = "visual"
        visual_mod = ".".join(package_split)
        visual_mod = importlib.import_module(visual_mod)
        logger.info(f"{visual_mod} is patched.")
        visual_mod.Attention.forward = attention_forward_visual

        # package_split[-1] = "cross_visual"
        # cross_visual_mod = ".".join(package_split)
        # cross_visual_mod = importlib.import_module(cross_visual_mod)
        # logger.info(f"{cross_visual_mod} is patched.")
        # cross_visual_mod.Attention.forward = attention_forward_cross_visual


@when_imported("transformers")
def patch_cogvlm_chat_hf(mod):
    if mod.__version__ != "4.40.2":
        # https://github.com/THUDM/GLM-4/issues/439
        # https://modelers.cn/models/openMind-ecosystem/cogagent-chat-hf
        logger.warning(
            f"when running cogvlm_chat_hf, please install transformers==4.40.2, but got: {
                mod.__version__}"
        )

    get_class_in_module_patched = patch_get_class_in_module(func=_patch_cogvlm_chat_hf)
    mod.dynamic_module_utils.get_class_in_module = get_class_in_module_patched
