import threading
import time

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig
from transformers import TextIteratorStreamer

from amp.evaluator.evaluator import Evaluator


class Inferencer(Evaluator):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._infer_config = cfg.inference

    def measure_performance(self):
        if self._infer_config.prompt is not None:
            prompt = self._infer_config.prompt
            input_ids = self._tokenizer.encode(prompt, return_tensors="pt").to(
                self._device
            )
        else:
            input_ids = torch.randint((1, 1024)).to(self._model.device)
        if self._prof:
            self._infer_config.warmup_runs = max(2, self._infer_config.warmup_runs)
        for i in range(self._infer_config.warmup_runs):
            with torch.no_grad():
                if self._prof and i == 0:
                    self._prof.start()
                self._model.generate(
                    input_ids, max_length=input_ids.shape[1] + 5, do_sample=False
                )
                if self._prof and i == 0:
                    self._prof.stop()

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
    i = Inferencer(cfg)
    i.measure_performance()


if __name__ == "__main__":
    model_inference()
