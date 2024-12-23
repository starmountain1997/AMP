import os
import threading
import time
from typing import Dict, List

import torch
from loguru import logger
from openmind import (AutoModelForCausalLM, AutoTokenizer,
                      is_torch_npu_available)
from transformers import TextIteratorStreamer


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
        self, device: str = None, warmup_runs: int = 1, total_tokens: int = 100
    ):
        if device:
            self._device = device
        elif is_torch_npu_available():
            self._device = "npu"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"
        logger.info(f"Running inference on {self._device}...")

        if not isinstance(warmup_runs, int) or warmup_runs < 0:
            raise ValueError("warmup_runs must be a non-negative integer.")
        self._warmup_runs = warmup_runs

        if not isinstance(total_tokens, int) or total_tokens < 0:
            raise ValueError("total_tokens must be a non-negative integer.")
        self._total_tokens = total_tokens

    def measure_performance(
        self,
        model_name: str,
        prompt: str|List[Dict] = None,
        tokenizer=None,
        if_chat: bool = False,
    ):
        if prompt is None:
            prompt=self.MESSAGE if if_chat else self.PROMPT
        if if_chat:
            prompt = ''.join(f"{msg['role'].capitalize()}: {msg['content']}\n" for msg in prompt)

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(self._device).eval()


        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self._device)

        for i in range(self._warmup_runs):
            logger.debug(f"Running {i+1}/{self._warmup_runs} warm-up inference...")
            with torch.no_grad():
                model.generate(
                    input_ids, max_length=input_ids.shape[1] + 5, do_sample=False
                )

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

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

    @classmethod
    def measure_performance_multimodal(
        cls,
        model_name: str,  # Replace with your multimodal model
        messages: List[Dict] = None,
        processor=None,
        tokenizer=None,
        device=None,
        total_tokens=50,
        warmup_runs=1,
        max_new_tokens=50,
    ):
        """
        Measure Time To First Token (TTFT) and Tokens Per Second (TPS) using streaming generation for multimodal models.

        :param image_path_or_url: Path or URL to the image file.
        :param audio_path_or_url: Path or URL to the audio file.
        :param prompt: Textual prompt to guide generation.
        :param model_name: Name of the Hugging Face multimodal model.
        :param processor_image: Preprocessor for image input.
        :param processor_audio: Preprocessor for audio input.
        :param tokenizer: Tokenizer for text input.
        :param device: 'cpu' or 'cuda'.
        :param total_tokens: Number of tokens to generate.
        :param warmup_runs: Number of warm-up inferences before measurement.
        :return: TTFT (seconds), TPS (tokens per second), generated text.
        """
        # Load tokenizer and model
        if device is None:
            device = "npu" if is_torch_npu_available() else "cuda"

        from openmind_hub import snapshot_download

        model_name = snapshot_download(model_name)

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        model.to(device).eval()

        if processor is None:
            processor = model._get_or_init_processor()

        if messages is None:
            messages = cls.MULTIMODAL_MESSAGE

        inputs = processor(messages)

        # Warm-up phase
        if warmup_runs > 0:
            for _ in range(warmup_runs):
                with torch.no_grad():
                    model.generate(
                        **inputs, max_new_tokens=max_new_tokens, do_sample=False
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
                    max_new_tokens=max_new_tokens,
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
            if len(generated_tokens) >= total_tokens:
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


if __name__ == "__main__":
    i = Inferencer()
    model_name_or_path = "openMind-ecosystem/Yi-6B"
    i.measure_performance(model_name_or_path)
    model_name_or_path = "openMind-ecosystem/Yi-1.5-9b-chat"
    i.measure_performance(model_name_or_path, if_chat=True)
    # model_name_or_path = "zyl9737/deepseek-coder-6.7b-instruct"
    # model_name_or_path = "zyl9737/deepseek-coder-6.7b-instruct_merge_lora"
    # Inferencer.measure_performance_chat(model_name_or_path)
    # model_name_or_path = "openMind-ecosystem/cogagent-chat-hf"
    # Inferencer.measure_performance_multimodal(model_name_or_path)
