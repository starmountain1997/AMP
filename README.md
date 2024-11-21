# AscendModelPatcher

## llama

```python
import time

from tqdm import tqdm
from transformers import pipeline


def main(use_amp=False):
    if use_amp:
        pass

    model_id = "meta-llama/Llama-3.2-1B"
    warmup_times = 10
    repeat_times = 10

    pipe = pipeline(
        "text-generation",
        model=model_id,
        device_map="auto"
    )

    for _ in tqdm(range(warmup_times), desc="Warming up"):
        pipe("The key to life is")

    start_time = time.time()
    for _ in tqdm(range(repeat_times), desc="Generating text"):
        pipe("The key to life is")
    end_time = time.time()

    print(
        f"Time taken: {(end_time - start_time) / repeat_times} seconds, use_amp: {use_amp}")


if __name__ == "__main__":
    main(use_amp=False)
```