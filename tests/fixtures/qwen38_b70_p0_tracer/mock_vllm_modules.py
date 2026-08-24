"""Tiny stand-in for vLLM rejection_sample used by offline hook tests."""
from __future__ import annotations

from types import SimpleNamespace

_calls: list[dict[str, object]] = []


class _Tensor:
    def __init__(self, values, *, argmax_dim: int | None = None) -> None:
        self._values = values
        self._argmax_dim = argmax_dim

    def detach(self) -> "_Tensor":
        return self

    def cpu(self) -> "_Tensor":
        return self

    def tolist(self):
        if self._argmax_dim is None:
            return self._values
        if self._argmax_dim == -1:
            return [row.index(max(row)) for row in self._values]
        raise NotImplementedError(self._argmax_dim)

    def argmax(self, dim: int) -> "_Tensor":
        return _Tensor(self._values, argmax_dim=dim)


def rejection_sample(
    draft_token_ids,
    draft_probs,
    target_logits,
    bonus_token_ids,
    num_draft_tokens,
    cu_num_draft_tokens,
    *,
    sampled_token_ids=None,
):
    _calls.append(
        {
            "draft_token_ids": draft_token_ids,
            "target_logits": target_logits,
            "bonus_token_ids": bonus_token_ids,
            "num_draft_tokens": num_draft_tokens,
            "cu_num_draft_tokens": cu_num_draft_tokens,
        }
    )
    if sampled_token_ids is not None:
        return sampled_token_ids
    return _Tensor([[101, 102, 999, -1, -1]])



class RejectionSampler:
    def forward(self, *args, **kwargs):
        return kwargs.get("sampled_token_ids")
class GPUModelRunner:
    def __init__(self) -> None:
        self.input_batch = SimpleNamespace(
            req_ids=["single-8k"],
            num_computed_tokens_cpu=_Tensor([8459]),
        )
        self.pending_rejection: dict[str, object] | None = None

    def _sample(self, *args, **kwargs):
        if self.pending_rejection is None:
            return None
        return rejection_sample(**self.pending_rejection)
