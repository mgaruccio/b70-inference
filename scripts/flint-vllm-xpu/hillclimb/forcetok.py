"""Teacher-forcing logits processor (vLLM Model Runner V2 interface).

A request with ``extra_args={'force': [ids...], 'prompt_len': P}`` has every row's
logits masked to the forced token at output index pos + 1 - P, so greedy decoding
replays the given sequence through the normal decode path while the engine still
reports raw logprobs (``logprobs_mode='raw_logprobs'``) of the forced tokens.
"""
from __future__ import annotations

import torch

from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsProcessor

LMAX = 4096


class ForceTokens(LogitsProcessor):
    def __init__(self, vllm_config, req_states):
        n = req_states.max_num_reqs
        dev = req_states.device
        self.force = torch.zeros(n, LMAX, dtype=torch.long, device=dev)
        self.plen = torch.zeros(n, dtype=torch.long, device=dev)
        self.flen = torch.ones(n, dtype=torch.long, device=dev)
        self.on = torch.zeros(n, dtype=torch.bool, device=dev)

    @classmethod
    def validate_params(cls, sp):
        ea = sp.extra_args or {}
        if 'force' in ea and not (isinstance(ea['force'], list) and len(ea['force']) <= LMAX):
            raise ValueError('force must be a list of token ids')

    def add_request(self, req_idx, sp):
        ea = sp.extra_args or {}
        f = ea.get('force')
        if f is None:
            self.on[req_idx] = False
            return False
        self.force[req_idx, :len(f)] = torch.tensor(f, dtype=torch.long)
        self.flen[req_idx] = len(f)
        self.plen[req_idx] = int(ea['prompt_len'])
        self.on[req_idx] = True
        return True

    def apply(self, logits, ctx):
        slots = ctx.expanded_idx_mapping.long()
        oi = (ctx.pos.long() + 1 - self.plen[slots]).clamp(min=0)
        oi = torch.minimum(oi, self.flen[slots] - 1)
        t = self.force[slots, oi]
        keep = torch.zeros_like(logits, dtype=torch.bool)
        keep.scatter_(1, t[:, None], True)
        logits.masked_fill_(~keep & self.on[slots][:, None], float('-inf'))
        return logits
