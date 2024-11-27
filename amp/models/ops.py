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


class NPUMinLengthLogitsProcessor(MinLengthLogitsProcessor):
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        vocab_tensor = torch.arange(scores.shape[-1], device=scores.device)
        eos_token_id = torch.tensor(self.eos_token_id, device=scores.device)
        eos_token_mask = npu_isin(vocab_tensor, eos_token_id)
        scores_processed = scores.clone()
        if input_ids.shape[-1] < self.min_length:
            scores_processed = torch.where(eos_token_mask, -math.inf, scores)
        return scores_processed


class NPUMinNewTokensLengthLogitsProcessor(MinNewTokensLengthLogitsProcessor):
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        new_tokens_length = input_ids.shape[-1] - self.prompt_length_to_skip
        scores_processed = scores.clone()
        vocab_tensor = torch.arange(scores.shape[-1], device=scores.device)
        eos_token_id = torch.tensor(self.eos_token_id, device=scores.device)
        eos_token_mask = npu_isin(vocab_tensor, eos_token_id)
        if new_tokens_length < self.min_new_tokens:
            scores_processed = torch.where(eos_token_mask, -math.inf, scores)

        return scores_processed


class NPUSuppressTokensAtBeginLogitsProcessor(
        SuppressTokensAtBeginLogitsProcessor):
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        vocab_tensor = torch.arange(scores.shape[-1], device=scores.device)
        begin_suppress_tokens = torch.tensor(
            self.begin_suppress_tokens, device=scores.device)
        suppress_token_mask = npu_isin(vocab_tensor, begin_suppress_tokens)
        scores_processed = scores
        if input_ids.shape[-1] == self.begin_index:
            scores_processed = torch.where(
                suppress_token_mask, -float("inf"), scores)

        return scores_processed


class NPUSuppressTokensLogitsProcessor(SuppressTokensLogitsProcessor):
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        vocab_tensor = torch.arange(scores.shape[-1], device=scores.device)
        suppress_tokens = torch.tensor(
            self.suppress_tokens, device=scores.device)
        suppress_token_mask = npu_isin(vocab_tensor, suppress_tokens)
        scores = torch.where(suppress_token_mask, -float("inf"), scores)
        return scores


class NPUEosTokenCriteria(EosTokenCriteria):
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(
            self,
            input_ids: torch.LongTensor,
            scores: torch.FloatTensor,
            **kwargs) -> torch.BoolTensor:
        if input_ids.device.type == "mps":
            # https://github.com/pytorch/pytorch/issues/77764#issuecomment-2067838075
            is_done = (
                input_ids[:, -1]
                .tile(self.eos_token_id.shape[0], 1)
                .eq(self.eos_token_id.unsqueeze(1).to(input_ids.device))
                .sum(dim=0)
                .bool()
                .squeeze()
            )
        else:
            is_done = npu_isin(
                input_ids[:, -1], self.eos_token_id.to(input_ids.device))
        return is_done


@when_imported("transformers")
def patch_isin(mod):
    # FIXME: torch_npu 930版本之前支持
    mod.generation.logits_process.MinLengthLogitsProcessor = NPUMinLengthLogitsProcessor
    mod.generation.logits_process.MinNewTokensLengthLogitsProcessor = NPUMinNewTokensLengthLogitsProcessor
    mod.generation.logits_process.SuppressTokensAtBeginLogitsProcessor = NPUSuppressTokensAtBeginLogitsProcessor
    mod.generation.logits_process.SuppressTokensLogitsProcessor = NPUSuppressTokensLogitsProcessor
    mod.generation.stopping_criteria.EosTokenCriteria = NPUEosTokenCriteria
