import math

import torch
from transformers.generation.logits_process import (
    LOGITS_PROCESSOR_INPUTS_DOCSTRING, MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor, SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor)
from transformers.generation.stopping_criteria import (
    STOPPING_CRITERIA_INPUTS_DOCSTRING, EosTokenCriteria)
from transformers.utils.doc import add_start_docstrings

from amp.module_patcher import when_imported

# 算子缺失
# https://gitee.com/ascend/pytorch/issues/I9JGYU?from=project-issue
# 930 之后的 torch_npu 版本是支持的


def npu_isin(
    elements: torch.Tensor,
    test_elements: torch.Tensor,
    *,
    assume_unique: bool = False,
    invert: bool = False
) -> torch.Tensor:
    """
    Mimics torch.isin for tensors of arbitrary dimensions.

    Args:
        elements (torch.Tensor): Input tensor to test for membership.
        test_elements (torch.Tensor): Values against which to test membership.
        assume_unique (bool, optional): If True, assumes both `elements` and `test_elements` contain unique items.
        invert (bool, optional): If True, inverts the membership test.

    Returns:
        torch.Tensor: Boolean tensor of the same shape as `elements`, where:
            - True indicates that the element is in `test_elements` (or not in `test_elements` if `invert` is True).
    """
    if not assume_unique:
        test_elements = torch.unique(test_elements)

    elements_flat = elements.flatten()
    test_elements_flat = test_elements.flatten()

    result_flat = (elements_flat.unsqueeze(-1) ==
                   test_elements_flat).any(dim=-1)

    if invert:
        result_flat = ~result_flat

    return result_flat.view(elements.shape)

