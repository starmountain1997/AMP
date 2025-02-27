<div align="center">English | <a href="./README_zh.md">中文</a></div>

# AnyModelPatcher

This project is a framework designed for patching Transformers models. It allows users to modify and enhance the functionality of various models seamlessly.

The patching mechanism utilized in this project is inspired by [PEP 369](https://peps.python.org/pep-0369/), which provides a standardized approach for modifying modules after they have been imported. This allows for greater flexibility and control over the behavior of the models.

For more information on modifying Transformers models, refer to the [How to Hack Any Transformers Model](https://github.com/huggingface/transformers/blob/main/docs/source/en/how_to_hack_models.md).

## installation

### install with Ascend

```bash
pip install git+https://github.com/starmountain1997/AMP.git#egg=AMP[ascend]
```

### local install

```shell
# use bash
pip install -e .[ascend]
# use zsh
pip install .\[ascend\]
```

## usage

### Inference Script Example

```python
import time

from tqdm import tqdm
from transformers import pipeline


def main(use_amp=False):
    if use_amp:
        from amp.models.llama import patch_llama

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

### LLaMA-Factory Training

#### train

To integrate the patching functionality into the training process, add the following line in `src/llamafactory/train/tuner.py`:

```
from amp.models.llama import patch_llama
```

## How to Patch?

For models that have been integrated into the `Transformers` repository, refer to [llama.py](amp/models/llama.py) for patching methods. For models that utilize dynamic loading, such as [cogagent-9b](https://huggingface.co/THUDM/cogagent-9b-20241220), follow the patching approach demonstrated in [cogagent2_9b.py](amp/models/cogagent2_9b.py). This ensures that the models can be modified effectively based on their loading mechanisms.
