import importlib
import math
import os
import sys
import types
from typing import Optional, Tuple

import av
import numpy as np
import torch
import torch_npu
from loguru import logger
from torch.nn import functional as F

from ..common.patch_transformers import patch_get_class_in_module
from ..module_patcher import when_imported
from .common.dummy_flash_attn import create_dummy_flash_attn
from .qwen2_vl import vision_flash_attention2_forward

# https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/ptmoddevg/trainingmigrguide/performance_tuning_0027.html


def flash_attention_forward_npu(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    seqlens: Optional[torch.LongTensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: bool = None,
):
    device = query_states.device

    if not use_top_left_mask:
        causal = is_causal
    else:
        causal = is_causal and query_length != 1

    atten_mask = (
        torch.from_numpy(np.triu(np.ones([2048, 2048]), k=1)).bool().to(device)
        if causal
        else None
    )  # FIXME:
    sparse_mode = 3 if causal else 2
    scale = (
        softmax_scale
        if softmax_scale is not None
        else 1.0 / math.sqrt(query_states.shape[-1])
    )

    if seqlens is not None:
        raise NotImplementedError
    elif attention_mask is not None:
        raise NotImplementedError
    else:
        attn_output = torch_npu.npu_fusion_attention(
            query_states,
            key_states,
            value_states,
            head_num=query_states.shape[2],
            input_layout="BSND",
            pse=None,
            keep_prob=1.0,
            scale=scale,
            atten_mask=atten_mask,
            sparse_mode=sparse_mode,
        )[0]

    return attn_output


def baichuan_whisper_attention_forward(
    self, hidden_states: torch.Tensor, seq_len: torch.Tensor
):
    bsz, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(bsz, self.num_heads, self.head_dim)
    key_states = self.k_proj(hidden_states).view(bsz, self.num_heads, self.head_dim)
    value_states = self.v_proj(hidden_states).view(bsz, self.num_heads, self.head_dim)

    cu_len = F.pad(torch.cumsum(seq_len, dim=0), (1, 0), "constant", 0).to(torch.int32)
    torch.max(seq_len).to(torch.int32).detach()

    head_num = query_states.shape[1]
    attn_output = torch_npu.npu_fusion_attention(
        query_states,
        key_states,
        value_states,
        head_num,
        pse=None,
        atten_mask=None,
        scale=1.0 / math.sqrt(query_states.shape[-1]),
        keep_prob=1,
        input_layout="TND",
        actual_seq_qlen=tuple(cu_len[1:].cpu().numpy().tolist()),
        actual_seq_kvlen=tuple(cu_len[1:].cpu().numpy().tolist()),
    )[0]
    attn_output = attn_output.reshape(bsz, self.embed_dim)
    attn_output = self.out_proj(attn_output)
    return attn_output


def baichuan_visual_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    causal_attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""

    bsz, tgt_len, embed_dim = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(
        bsz * tgt_len, self.num_heads, self.head_dim
    )
    key_states = self.k_proj(hidden_states).view(
        bsz * tgt_len, self.num_heads, self.head_dim
    )
    value_states = self.v_proj(hidden_states).view(
        bsz * tgt_len, self.num_heads, self.head_dim
    )

    # 暂时不考虑变长patch nums 固定长度为256/1024
    cu_len = torch.arange(
        0,
        (bsz + 1) * tgt_len,
        step=tgt_len,
        dtype=torch.int32,
        device=query_states.device,
    )
    # print(self.config.s2a, self.config.rope_scaling, cu_len, torch.sum(cu_len), q_len, kv_seq_len)
    # 如果不是f16 bf16不用flash attn
    if query_states.dtype in [torch.float16, torch.bfloat16]:
        head_num = query_states.shape[1]
        attn_output = torch_npu.npu_fusion_attention(
            query_states,
            key_states,
            value_states,
            head_num,
            pse=None,
            atten_mask=attention_mask,
            scale=1.0 / math.sqrt(query_states.shape[-1]),
            keep_prob=1,
            input_layout="TND",
            actual_seq_qlen=tuple(cu_len[1:].cpu().numpy().tolist()),
            actual_seq_kvlen=tuple(cu_len[1:].cpu().numpy().tolist()),
        )
        attn_output = attn_output.view(bsz, tgt_len, self.num_heads, self.head_dim)
    else:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False
        ):
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attention_mask, 0.0
            )
            attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output = self.out_proj(attn_output)

    return attn_output, None


@torch.no_grad()
def baichuan_visual_encoder_fake_input(self, device):
    merge_size = max(self.merge_size, self.config.spatial_merge_size)
    flatten_patches = torch.zeros(
        (
            merge_size * merge_size,
            3 * self.config.temporal_patch_size * self.config.patch_size**2,
        ),
        dtype=torch.float32,
        device=device,
    )
    return [flatten_patches], [(1, merge_size, merge_size)], [1]


def read_video_pyav(image_path, max_frame_number, decode_way):
    if decode_way == "1fps":
        try:
            with av.open(image_path) as container:
                stream = container.streams.video[0]
                fps = int(stream.average_rate)

                frames = []
                frame_times = []
                cnt = 0

                for frame in container.decode(stream):
                    if cnt % fps == 0:  # Extract 1 frame per second
                        image = np.array(frame.to_image())
                        frames.append(image)
                        frame_time = int(frame.time * 1000)  # Convert to milliseconds
                        frame_times.append(frame_time)
                    cnt += 1

        except Exception as e:
            print(image_path)
            print("error is", e)
            return None

    elif decode_way == "key":
        try:
            with av.open(image_path) as container:
                stream = container.streams.video[0]
                stream.codec_context.skip_frame = "NONKEY"

                frames = []
                frame_times = []

                for frame in container.decode(stream):
                    image = np.array(frame.to_image())
                    frames.append(image)
                    # Convert to milliseconds
                    frame_time = int(frame.time * 1000)
                    frame_times.append(frame_time)

        except Exception as e:
            print("error is", e)
            return None

    else:
        print("Invalid decode_way specified")
        return None

    # If no frames are extracted, return None
    if not frames:
        return None

    # If the number of frames exceeds max_frame_number, uniformly sample frames
    if len(frames) > max_frame_number > 0:
        indices = np.linspace(0, len(frames) - 1, max_frame_number, dtype=int)
        frames = [frames[i] for i in indices]
        frame_times = [frame_times[i] for i in indices]

    return frames, frame_times


def _patch_bc5_14b_omini(mod):
    package_name = mod.__name__.split(".")[-1]

    decord = types.ModuleType("decord")
    sys.modules["decord"] = decord
    decord.cpu = None
    decord.VideoReader = None

    if package_name == "modeling_baichuan":
        logger.info(f"{mod} is patched.")
        mod.flash_attention_forward = flash_attention_forward_npu

        package_split = mod.__name__.split(".")
        package_split[-1] = "audio_modeling_baichuan"
        audio_modeling_baichuan_mod = ".".join(package_split)
        audio_modeling_baichuan_mod = importlib.import_module(
            audio_modeling_baichuan_mod
        )
        logger.info(f"{audio_modeling_baichuan_mod} is patched.")
        audio_modeling_baichuan_mod.BaichuanWhisperAttention.forward = (
            baichuan_whisper_attention_forward
        )

        package_split[-1] = "visual_modeling_baichuan"
        visual_modeling_baichuan_mod = ".".join(package_split)
        visual_modeling_baichuan_mod = importlib.import_module(
            visual_modeling_baichuan_mod
        )
        logger.info(f"{visual_modeling_baichuan_mod} is patched.")
        visual_modeling_baichuan_mod.BaichuanVisualAttention.forward = (
            baichuan_visual_attention_forward
        )
        visual_modeling_baichuan_mod.BaichuanVisualEncoder.fake_input = (
            baichuan_visual_encoder_fake_input
        )

        package_split[-1] = "processor_baichuan"

        processor_baichuan_mod = ".".join(package_split)
        processor_baichuan_mod = importlib.import_module(processor_baichuan_mod)
        processor_baichuan_mod.read_video = read_video_pyav

        # package_split[-1] = "sequence_parallel_utils"
        # sequence_parallel_utils_mod = ".".join(package_split)
        # sequence_parallel_utils_mod = importlib.import_module(
        #     sequence_parallel_utils_mod
        # )
        # logger.info(f"{sequence_parallel_utils_mod} is patched.")
        # sequence_parallel_utils_mod.LocalAttention.forward = local_attention_forward


@when_imported("transformers")
def patch_bc5_7b_omini(mod):
    if mod.__version__ != "4.47.1":
        logger.warning(
            f"when running cogvlm_chat_hf, please install transformers==4.47.1, but got: {
                mod.__version__}"
        )
    os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"
    create_dummy_flash_attn()

    mod.modeling_flash_attention_utils._flash_supports_window_size = None
    mod.modeling_flash_attention_utils._upad_input = None
    mod.modeling_flash_attention_utils.prepare_fa2_from_position_ids = None
    mod.utils.is_flash_attn_2_available = lambda: True
    mod.modeling_utils.is_flash_attn_2_available = lambda: True  # 相对路径引入
    mod.utils.is_flash_attn_greater_or_equal_2_10 = lambda: True
    mod.models.qwen2_vl.modeling_qwen2_vl.VisionFlashAttention2.forward = (
        vision_flash_attention2_forward
    )

    get_class_in_module_patched = patch_get_class_in_module(func=_patch_bc5_14b_omini)
    mod.dynamic_module_utils.get_class_in_module = get_class_in_module_patched
