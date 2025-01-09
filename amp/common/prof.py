import functools
import os
import os.path as osp
import platform
import re
import subprocess
import threading
import time
from contextlib import contextmanager
from datetime import datetime

import torch
from loguru import logger
from openmind import is_torch_npu_available
from torch.profiler import record_function
from transformers import (PreTrainedModel, PreTrainedTokenizer,
                          TextIteratorStreamer)


def _find_latest_prof_folder(base_path: str) -> str:
    folder_pattern = re.compile(
        rf"^{platform.node()}_([0-9]+)_([0-9]{{17}})_ascend_pt$"
    )
    latest_folder = None
    latest_time = None
    for folder_name in os.listdir(base_path):
        match = folder_pattern.match(folder_name)
        if match:
            utc_time = match.group(2)
            try:
                utc_dt = datetime.strptime(utc_time, "%Y%m%d%H%M%S%f")

                if latest_time is None or utc_dt > latest_time:
                    latest_time = utc_dt
                    latest_folder = folder_name
            except ValueError:
                print(f"Skipping invalid folder name: {folder_name}")
                continue

    if latest_folder:
        return latest_folder
    raise ValueError(f"No valid folder found under the base path: {base_path}.")


def get_profiler(device: str = None, save_path="./"):
    if device:
        pass
    elif is_torch_npu_available():
        device = "npu"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if device.startswith("npu"):
        import torch_npu

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
            l2_cache=False,
        )
        prof = torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(save_path),
            record_shapes=False,  # 大模型膨胀较多
            profile_memory=False,
            with_stack=False,  # 大模型膨胀较多，但是堆栈信息是分析必须的
            with_flops=False,
            with_modules=False,
            experimental_config=experimental_config,
        )

    elif device.startswith("cuda"):
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(save_path),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
        )
    else:
        raise NotImplementedError
    return prof


def run_advisor(prof_save_path: str = "./"):
    """
    运行advisor
    """
    prof_folder = _find_latest_prof_folder(prof_save_path)
    command = [
        "msprof-analyze",
        "advisor",
        "all",
        "-d",
        "ASCEND_PROFILER_OUTPUT",
    ]
    logger.info(" ".join(command))
    subprocess.run(command, cwd=os.path.join(prof_save_path, prof_folder))


def measure_performance(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    total_tokens: int,
    input_kwargs,
):
    input_ids = input_kwargs.get("input_ids")
    input_kwargs["max_length"] = input_ids.shape[1] + total_tokens

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    generated_tokens = []
    ttft_recorded = False
    ttft = None

    def generate():
        with torch.no_grad():
            model.generate(
                streamer=streamer,
                **input_kwargs,
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

        if len(generated_tokens) >= total_tokens:
            break

    end_time = time.perf_counter()

    thread.join()

    total_time = end_time - start_time
    tps = len(generated_tokens) / total_time if total_time > 0 else float("inf")

    generated_text = "".join(generated_tokens)
    logger.info(
        f"TTFT: {ttft:.3f} s, TPS: {tps:.3f} tokens/s.\n\nresponse:\n{generated_text}"
    )

    return ttft, tps, generated_text


def run_compare(task_1: str, task_2: str, save_path: str):
    """
    https://gitee.com/ascend/mstt/tree/master/profiler/compare_tools
    比较
    """
    task_1_prof_folder = _find_latest_prof_folder(task_1)
    task_2_prof_folder = _find_latest_prof_folder(task_2)
    command = [
        "msprof-analyze",
        "compare",
        "-d",
        osp.join(task_1_prof_folder, "ASCEND_PROFILER_OUTPUT"),
        "-d",
        osp.join(task_2_prof_folder, "ASCEND_PROFILER_OUTPUT"),
        "--output_path=./",
    ]
    logger.info(" ".join(command))
    subprocess.run(command, cwd=save_path)


def torch_profiler_wrapper(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Initialize the profiler
        prof = get_profiler()
        with prof:
            with record_function("forward"):
                result = func(*args, **kwargs)
            prof.step()  # Advance the profiler step
        return result

    return wrapper


@contextmanager
def patch_forward_with_profiler(model):
    original_forward = model.forward
    model.forward = torch_profiler_wrapper(original_forward)
    try:
        yield
    finally:
        model.forward = original_forward


@contextmanager
def patch_generate_with_profiler(model):
    original_generate = model.generate
    model.generate = torch_profiler_wrapper(original_generate)
    try:
        yield
    finally:
        model.generate = original_generate
