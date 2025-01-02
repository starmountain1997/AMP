import os.path as osp
import threading
import time
from typing import Dict, List

import hydra
import torch
import torch_npu
from loguru import logger
from omegaconf import DictConfig
from openmind import is_torch_npu_available
from openmind_hub import snapshot_download
from transformers import TextIteratorStreamer
import importlib


class Inferencer:
    def __init__(self,cfg: DictConfig):
        inferencer_cfg=cfg.inferencer

        if is_torch_npu_available():
            self._device = "npu"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"


        if inferencer_cfg.profiler and inferencer_cfg.warmup_runs < 2:
            logger.warning("warmup_runs must be a positive integer when need_profiling is True.")
            self._warmup_runs = max(2, inferencer_cfg.warmup_runs)
        elif inferencer_cfg.warmup_runs < 0:
            raise ValueError("warmup_runs must be a non-negative integer.")
        else:
            self._warmup_runs = inferencer_cfg.warmup_runs

        if not isinstance(inferencer_cfg.total_tokens, int) or inferencer_cfg.total_tokens < 0:
            raise ValueError("total_tokens must be a non-negative integer.")
        self._total_tokens = int(inferencer_cfg.total_tokens)


        self._model = self.set_model(inferencer_cfg.model)
        self._tokenizer = self.set_tokenizer(inferencer_cfg.tokenizer)
        if inferencer_cfg.profiler:
            self._profiler = self.set_profiler(inferencer_cfg.profiling_save_path)
        else:
            self._profiler = None

    def set_profiler(self, profiling_save_path: str):
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
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    profiling_save_path
                ),
                record_shapes=False,  # 大模型膨胀较多
                profile_memory=True,
                with_stack=True,  # 大模型膨胀较多，但是堆栈信息是分析必须的
                with_flops=False,
                with_modules=False,
                experimental_config=experimental_config,
            )
        else:
            # TODO: CPU/GPU profiling
            raise NotImplementedError("Profiler for CPU/GPU is not implemented yet.")
        return prof

    def set_model(
        self,
        model_cfg: DictConfig,
    ):
        if model_cfg.platform == "modelers":
            model_name = snapshot_download(model_cfg.name,
                                           revision="npu")
        else:
            model_name = model_cfg.name
        # load_class = importlib.import_module(f"transformers.{model_cfg.load_class}")
        load_class=transformers.getattr(model_cfg.load_class)
        model = load_class.from_pretrained(
            model_name,
            device_map=model_cfg.device_map,
            torch_dtype=model_cfg.torch_dtype,
            trust_remote_code=True,
        ).eval()
        return model

    def set_tokenizer(self,  tokenizer_cfg: DictConfig,):
        if tokenizer_cfg.platform == "modelers":
            model_name = snapshot_download(tokenizer_cfg.name)
        else:
            model_name = tokenizer_cfg.name
        # load_class = importlib.import_module(f"transformers.{tokenizer_cfg.load_class}")
        load_class=transformers.getattr(tokenizer_cfg.load_class)

        tokenizer = load_class.from_pretrained(
            model_name, trust_remote_code=True
        )
        return tokenizer

    def measure_performance(
        self,
    ):


        
        # input_ids = self._tokenizer.encode(prompt, return_tensors="pt").to(self._device)
        input_ids=torch.randint((1,1024)).to(self._model.device)

        for i in range(self._warmup_runs):
        #     if self._need_profiling and i == 0:
        #         prof_save_path = osp.join(
        #             self._profiling_save_path,
        #             f"profiling_{model_name.replace('/','_')}",
        #         )
        #         prof = self._get_profiler(prof_save_path)
        #         logger.info(
        #             f"Running {i+1}/{self._warmup_runs} warm-up inference... profiling data save to {prof_save_path}"
        #         )
        #     else:
        #         prof = None
        #         logger.info(f"Running {i+1}/{self._warmup_runs} warm-up inference...")

            with torch.no_grad():
                # if prof:
                #     prof.start()
                self._model.generate(
                    input_ids, max_length=input_ids.shape[1] + 5, do_sample=False
                )
                # if prof:
                #     prof.stop()

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        generated_tokens = []
        ttft_recorded = False
        ttft = None

        def generate():
            with torch.no_grad():
                self._model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + self._total_tokens,
                    do_sample=False,
                    streamer=streamer,
                )

        thread = threading.Thread(target=generate)
        thread.start()

        start_time = time.perf_counter()

        for token in streamer:
            current_time = time.perf_counter()
            generated_tokens.append(token)
            elapsed_time = current_time - start_time

            if not ttft_recorded:
                ttft = elapsed_time
                ttft_recorded = True

            if len(generated_tokens) >= self._total_tokens:
                break

        end_time = time.perf_counter()

        thread.join()

        total_time = end_time - start_time
        tps = len(generated_tokens) / total_time if total_time > 0 else float("inf")

        generated_text = "".join(generated_tokens)
        logger.info(f"TTFT: {ttft:.3f} s, TPS: {tps:.3f} tokens/s.")
        # logger.info(f"prompt: {prompt}\nresponse: {generated_text}")

        return ttft, tps, generated_text


@hydra.main(config_path=".", config_name="cogagent-chat-hf")
def model_inference(cfg: DictConfig) -> None:
    i = Inferencer(
        cfg
    )
    
    i.measure_performance(
    )


if __name__ == "__main__":
    model_inference()
