# llama

## 融合算子

```
from amp.models.llama import patch_llama
```

# qwen

## 融合算子

```
from amp.models.qwen import patch_qwen
```

# chatglm
## `torch.isin`

2024.9.30 之前的 torch_npu 版本不支持 `torch.isin`，建议使用 monkey-patch 替换：
```python
from amp.models.ops import npu_isin

torch.isin = npu_isin
``` 

## 融合算子

```
from amp.models.chatglm import patch_chatglm
```

# cogvlm
- triton 尚不支持
- xformers 尚不支持
