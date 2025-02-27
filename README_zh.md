<div align="center"><a href="./README.md">English</a> | 中文</div>

# AnyModelPatcher

该项目是旨在修补 Transformers 模型的框架，它允许用户无缝地修改和增强各种模型的功能。

该项目中使用的修补机制灵感来源于 [PEP 369](https://peps.python.org/pep-0369/)，它提供了一种标准化的方法，用于在模块导入后对其进行修改。这使得用户能够更灵活地控制模型的行为。

有关如何修改 Transformers 模型的更多信息，请参阅 [How to Hack Any Transformers Model](https://github.com/huggingface/transformers/blob/main/docs/source/en/how_to_hack_models.md)。

## 安装

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

## 使用

### 推理脚本示例

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

### LLaMA-Factory 训练

#### train

要将修补功能集成到训练过程中，请在 `src/llamafactory/train/tuner.py` 中添加以下行：

```
from amp.models.llama import patch_llama
```

## 如何修补？

对于已经集成到 `Transformers` 仓库中的模型，请参考 [llama.py](amp/models/llama.py) 中的修补方法。对于使用动态加载的模型，例如 [cogagent-9b](https://huggingface.co/THUDM/cogagent-9b-20241220)，请遵循 [cogagent2_9b.py](amp/models/cogagent2_9b.py) 中演示的修补方法。这确保了可以根据模型的加载机制有效地对其进行修改。
