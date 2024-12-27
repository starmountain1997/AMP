import os
import threading
import time
from typing import Callable, Dict, List

import torch
from loguru import logger
from openmind import (AutoModelForCausalLM, AutoProcessor, AutoTokenizer,
                      is_torch_npu_available)
from PIL import Image
from transformers import TextIteratorStreamer
import torch_npu
from utils import run_advisor
import os.path as osp
# from amp.models.yi import patch_yi

class Inferencer:
    PROMPT = "Can you give me some advice on how to keep healthy?"
    MESSAGE = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! Can you help me with my homework?"},
        {"role": "assistant", "content": "Of course! What subject are you working on?"},
    ]

    MULTIMODAL_MESSAGE = [
        {
            "role": "user",
            "content": {
                "image": os.path.join(os.path.dirname(__file__), "test_image.png"),
                # "audio":os.path.join(os.path.dirname(__file__), "test_audio.m4a"),
                "text": "Please describe the content of the image.",
            },
        },
    ]

    def __init__(
        self, device: str = None, warmup_runs: int = 1, total_tokens: int = 100,
        need_profiling: bool = False,
        profiling_save_path: str = "./",
    ):
        if device:
            self._device = device
        elif is_torch_npu_available():
            self._device = "npu"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"
        if need_profiling and warmup_runs == 0:
            logger.warning("warmup_runs must be a positive integer when need_profiling is True.")
            warmup_runs = 1

        logger.info(f"Running inference on {self._device}...")

        if not isinstance(warmup_runs, int) or warmup_runs < 0:
            raise ValueError("warmup_runs must be a non-negative integer.")
        self._warmup_runs = warmup_runs

        if not isinstance(total_tokens, int) or total_tokens < 0:
            raise ValueError("total_tokens must be a non-negative integer.")
        self._total_tokens = total_tokens

        self._need_profiling = need_profiling
        self._profiling_save_path = profiling_save_path

    @staticmethod
    def _get_profiler(device: str, prof_save_path: str):
        logger.info(f"Profiling inference on {device}... save to {prof_save_path}")
        if device == "npu":
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
                    prof_save_path
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=False,
                with_modules=False,
                experimental_config=experimental_config,
            )
        else:
            raise NotImplementedError("Profiler for CPU is not implemented yet.")
        return prof

    def measure_performance(
        self,
        model_name: str,
        prompt: str | List[Dict] = None,
        tokenizer=None,
        if_chat: bool = False,
    ):
        if prompt is None:
            prompt = self.MESSAGE if if_chat else self.PROMPT
        if if_chat:
            prompt = "".join(
                f"{msg['role'].capitalize()}: {msg['content']}\n" for msg in prompt
            )

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
        ).eval()

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self._device)

        for i in range(self._warmup_runs):
            logger.debug(f"Running {i+1}/{self._warmup_runs} warm-up inference...")
            prof_save_path=osp.join(self._profiling_save_path,f"profiling_{model_name.replace('/','_')}_{i}")
            prof=self._get_profiler(device=self._device,prof_save_path=prof_save_path) if self._need_profiling else None
            with torch.no_grad():
                if prof:
                    logger.debug("Profiling warm-up inference...")
                    prof.start()
                model.generate(
                    input_ids, max_length=input_ids.shape[1] + 5, do_sample=False
                )
                if prof:
                    prof.stop()

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        generated_tokens = []
        ttft_recorded = False
        ttft = None

        def generate():
            with torch.no_grad():
                model.generate(
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
        logger.info(f"prompt: {prompt}\nresponse: {generated_text}")

        return ttft, tps, generated_text

    def measure_performance_multimodal(
        self,
        model_name: str,  # Replace with your multimodal model
        processor: Callable = None,
        messages: List[Dict] = None,
        tokenizer=None,
    ):
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
        if isinstance(model_name, str):
            model = (
                AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                )
                .to(self._device)
                .eval()
            )
        else:
            model = model_name

        if messages is None:
            messages = self.MULTIMODAL_MESSAGE

        if processor is None:
            processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            inputs = processor(messages)
        else:
            inputs = processor(messages, model, tokenizer)

        # Warm-up phase
        for _ in range(self._warmup_runs):
            with torch.no_grad():
                model.generate(
                    **inputs,
                    max_length=inputs["input_ids"].shape[1] + 5,
                    do_sample=False,
                )

        # Initialize the streamer
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        # Shared variables to track performance
        generated_tokens = []
        ttft = None
        generation_started = False

        # Function to run model.generate in a separate thread
        def generate():
            with torch.no_grad():
                model.generate(
                    **inputs,
                    max_length=inputs["input_ids"].shape[1] + self._total_tokens,
                    do_sample=False,
                    streamer=streamer,
                )

        # Start generation in a separate thread
        thread = threading.Thread(target=generate)
        thread.start()

        # Start timer
        start_time = time.perf_counter()

        # Iterate over generated tokens
        for token in streamer:
            current_time = time.perf_counter()
            generated_tokens.append(token)

            # Record TTFT when the first token is received
            if not generation_started:
                ttft = current_time - start_time
                generation_started = True

            # Stop after receiving the desired number of tokens
            if len(generated_tokens) >= self._total_tokens:
                break

        # End timer
        end_time = time.perf_counter()

        # Clean up
        thread.join()

        # Calculate TPS
        total_time = end_time - start_time
        tps = len(generated_tokens) / total_time if total_time > 0 else float("inf")

        # Decode generated tokens
        generated_text = "".join(generated_tokens)
        logger.info(f"TTFT: {ttft:.3f} s, TPS: {tps:.3f} tokens/s.")
        logger.info(f"messages: {messages}\nresponse: {generated_text}")

        return ttft, tps, generated_text


def cogagent2_9b_processor(messages, model, tokenizer):
    last_user_input = [msg for msg in messages if msg["role"] == "user"][-1]
    image = last_user_input["content"]["image"]
    image = Image.open(image).convert("RGB")
    query = last_user_input["content"]["text"]
    input_by_model = model.build_conversation_input_ids(
        tokenizer, query=query, history=[], images=[image]
    )

    inputs = {
        "input_ids": input_by_model["input_ids"].unsqueeze(0).to("npu"),
        "token_type_ids": input_by_model["token_type_ids"].unsqueeze(0).to("npu"),
        "attention_mask": input_by_model["attention_mask"].unsqueeze(0).to("npu"),
        "images": [[input_by_model["images"][0].to("npu").to(torch.float16)]],
    }
    if "cross_images" in input_by_model and input_by_model["cross_images"]:
        inputs["cross_images"] = [
            [input_by_model["cross_images"][0].to("npu").to(torch.float16)]
        ]
    return inputs

def bunny_llama3_8b(messages,model,tokenizer):
    # https://github.com/starmountain1997/AMP.git
    pass

if __name__ == "__main__":
    i = Inferencer(need_profiling=True)

    # model = "ccpower/Bunny-Llama-3-8B-V"
    # model="zyl9737/deepseek-coder-6.7b-instruct"
    # model="libo2024/Yi-9B-200K"
    # model="ltdog/Qwen1.5-32B"
    # i.measure_performance(model_name=model)
    run_advisor(prof_save_path="/home/guozr/CODE/AMP/profiling_ltdog_Qwen1.5-32B_0")
