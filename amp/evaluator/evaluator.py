import importlib
import os.path as osp

import torch
import torch_npu
from loguru import logger
from omegaconf import DictConfig
from openmind import is_torch_npu_available
from openmind_hub import snapshot_download


class Evaluator:
    def __init__(self, cfg: DictConfig):
        if is_torch_npu_available():
            self._device = "npu"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"
        logger.info(f"Using device: {self._device}")

        self._model = self._set_model(cfg.model)
        self._tokenizer = self._set_tokenizer(cfg.tokenizer)

        if cfg.profiler:
            self._prof = self._set_profiler(cfg.profiler)
        else:
            self._prof = None

    def _set_profiler(self, prof_cfg: DictConfig):
        save_path = osp.join(
            prof_cfg.save_path,
            f"profiling_{self._model.config.name_or_path.replace('/', '_')}",
        )
        logger.info(f"profiling save path: {save_path}")
        if self._device == "npu":
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                l2_cache=False,
            )
            prof = torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(save_path),
                record_shapes=prof_cfg.record_shapes,  # 大模型膨胀较多
                profile_memory=prof_cfg.profile_memory,
                with_stack=prof_cfg.with_stack,  # 大模型膨胀较多，但是堆栈信息是分析必须的
                with_flops=prof_cfg.with_flops,
                with_modules=prof_cfg.with_modules,
                experimental_config=experimental_config,
            )
        else:
            prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(save_path),
                record_shapes=prof_cfg.record_shapes,
                profile_memory=prof_cfg.profile_memory,
                with_stack=prof_cfg.with_stack,
                with_flops=prof_cfg.with_flops,
                with_modules=prof_cfg.with_modules,
            )
        return prof

    def _set_model(
        self,
        model_cfg: DictConfig,
    ):
        if model_cfg.platform == "modelers":
            transformers_module = importlib.import_module("openmind")
        else:
            transformers_module = importlib.import_module("transformers")
        load_class = getattr(transformers_module, model_cfg.load_class)
        model = load_class.from_pretrained(
            model_cfg.name,
            device_map=model_cfg.device_map,
            torch_dtype=model_cfg.torch_dtype,
            trust_remote_code=True,
        ).eval()
        return model

    def _set_tokenizer(
        self,
        tokenizer_cfg: DictConfig,
    ):
        if tokenizer_cfg.platform == "modelers":
            model_name = snapshot_download(tokenizer_cfg.name)
        else:
            model_name = tokenizer_cfg.name
        transformers_module = importlib.import_module("transformers")
        load_class = getattr(transformers_module, tokenizer_cfg.load_class)

        tokenizer = load_class.from_pretrained(model_name, trust_remote_code=True)
        return tokenizer
