import torch
import torch.nn.functional as F
import torch_npu
from loguru import logger

from ..common.patch_transformers import patch_get_class_in_module
from ..module_patcher import when_imported


def patch_apply_rotary_pos_emb(
    x: torch.Tensor, rope_cache: torch.Tensor
) -> torch.Tensor:
    # x: [sq, b, np, hn]
    sq, b, np, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:sq]
    xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
    rope_cache = rope_cache.view(sq, -1, 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)


def patch_rms_norm_forward(self, hidden_states: torch.Tensor):
    input_dtype = hidden_states.dtype
    normalized_hidden_states, _ = torch_npu.npu_rms_norm(
        hidden_states, self.weight, epsilon=self.eps
    )
    return normalized_hidden_states.to(input_dtype)


def patch_core_attention_forward(
    self, query_layer, key_layer, value_layer, attention_mask
):
    pytorch_major_version = int(torch.__version__.split(".")[0])
    if pytorch_major_version >= 2:
        query_layer, key_layer, value_layer = [
            k.permute(1, 2, 0, 3) for k in [query_layer, key_layer, value_layer]
        ]
        if attention_mask is None and query_layer.shape[2] == key_layer.shape[2]:
            context_layer = torch.nn.functional.scaled_dot_product_attention(
                query_layer, key_layer, value_layer, is_causal=True
            )
        else:
            if attention_mask is not None:
                attention_mask = ~attention_mask
            context_layer = torch.nn.functional.scaled_dot_product_attention(
                query_layer, key_layer, value_layer, attention_mask
            )

        context_layer = torch_npu.npu_confusion_transpose(
            context_layer,
            (2, 0, 1, 3),
            (
                context_layer.size()[2],
                context_layer.size()[0],
                self.hidden_size_per_partition,
            ),
            transpose_first=True,
        )

    else:
        # Raw attention scores

        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(
            output_size[2], output_size[0] * output_size[1], -1
        )
        # [sk, b, np, hn] -> [sk, b * np, hn]
        # key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], output_size[2])

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
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            torch_npu.npu_confusion_transpose(
                key_layer,
                (1, 0, 2, 3),
                (output_size[3], output_size[0], output_size[1], output_size[2]),
                transpose_first=False,
            ).traspose(1, 2),
            # key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
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
            attention_scores = attention_scores.masked_fill(
                attention_mask, float("-inf")
            )
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = attention_probs.type_as(value_layer)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)
        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (
            value_layer.size(1),
            value_layer.size(2),
            query_layer.size(0),
            value_layer.size(3),
        )
        # change view [sk, b * np, hn]
        value_layer = value_layer.view(
            value_layer.size(0), output_size[0] * output_size[1], -1
        )
        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(
            output_size[0] * output_size[1], output_size[2], -1
        )
        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))
        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size_per_partition,
        )
        context_layer = context_layer.view(*new_context_layer_shape)

    return context_layer


def _patch_chatglm2_32k(mod, name):
    package_name = mod.__name__.split(".")[-1]
    if package_name == "modeling_chatglm":
        logger.info(f"{mod} is patched.")
        mod.RMSNorm.forward = patch_rms_norm_forward
        # mod.CoreAttention.forward = patch_core_attention_forward
        mod.apply_rotary_pos_emb = patch_apply_rotary_pos_emb
        mod.MLP.activation_func = lambda x: torch_npu.npu_swiglu(x, dim=-1)


@when_imported("transformers")
def patch_chatglm2_32k(mod):
    if mod.__version__ != "4.30.2":
        logger.warning(
            f"when running characterglm_6b, please install transformers==4.30.2, but got: {mod.__version__}"
        )

    get_class_in_module_patched = patch_get_class_in_module(func=_patch_chatglm2_32k)
    mod.dynamic_module_utils.get_class_in_module = get_class_in_module_patched
