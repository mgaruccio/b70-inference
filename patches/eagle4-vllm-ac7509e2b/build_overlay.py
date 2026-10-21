#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproduce the surgical engine patch from the two immutable reference trees.

Source changes are backported from Eagle's vLLM 3ba23801e80d2a0d0a9af21ce94484aa2adb8795
onto the exported B70 package 0.27.2rc1.dev77+gac7509e2b. No source tree is modified.
Only the two new generic hook modules are copied in full. Existing engine files
are patched by focused hunks; unrelated features of the newer fork are excluded.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

OFFICIAL = "3ba23801e80d2a0d0a9af21ce94484aa2adb8795"
IMAGE = "vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f"
VERSION = "0.27.2rc1.dev77+gac7509e2b"
HERE = Path(__file__).resolve().parent


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def symbol(source: str, name: str) -> str:
    node = ast.parse(source)
    for part in name.split("."):
        node = next(n for n in node.body if getattr(n, "name", None) == part)
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return "".join(source.splitlines(keepends=True)[start - 1:node.end_lineno])


def build(base: Path, official: Path) -> None:
    head = subprocess.check_output(["git", "-C", str(official.parent), "rev-parse", "HEAD"], text=True).strip()
    if head != OFFICIAL:
        raise ValueError(f"Wrong official reference: {head}")
    originals: dict[str, str] = {}
    changed: dict[str, str] = {}

    def load(path):
        originals[path] = (base / path).read_text()
        changed[path] = originals[path]

    def edit(path, old, new, count=1):
        if path not in changed:
            load(path)
        if changed[path].count(old) != count:
            raise ValueError(f"Expected {count} exact source matches in {path}: {old[:100]!r}")
        changed[path] = changed[path].replace(old, new)

    reference_cache = {}

    def reference(path):
        if path not in reference_cache:
            text = (official / path).read_text()
            committed = subprocess.check_output(
                ["git", "-C", str(official.parent), "show", f"{OFFICIAL}:vllm/{path}"], text=True)
            if text != committed:
                raise ValueError(f"Dirty official reference: {path}")
            reference_cache[path] = text
        return reference_cache[path]

    def upstream(path, name):
        return symbol(reference(path), name)

    for path in ("v1/spec_decode/registry.py", "model_executor/layers/mamba/gdn/verify_extension.py"):
        if (base / path).exists():
            raise ValueError(f"Expected new module: {path}")
        originals[path] = ""
        changed[path] = reference(path)

    p = "config/speculative.py"
    edit(p, "from vllm.config.utils import config", "from vllm.config.utils import config, normalize_value")
    edit(p, "from vllm.v1.attention.backends.registry import AttentionBackendEnum\n",
         "from vllm.v1.attention.backends.registry import AttentionBackendEnum\n"
         "from vllm.v1.spec_decode.registry import (\n"
         "    SpeculativeMethodSpec, get_speculative_method, registered_speculative_methods,\n)\n")
    edit(p, "    method: SpeculativeMethod | None = None", "    method: SpeculativeMethod | str | None = None")
    edit(p, "    draft_tensor_parallel_size: int | None = Field(default=None, ge=1)",
         "    method_config: dict[str, Any] = Field(default_factory=dict)\n"
         '    """Opaque plugin options, interpreted only by its configure hook/speculator."""\n'
         "    draft_tensor_parallel_size: int | None = Field(default=None, ge=1)")
    edit(p, '            "dspark",\n        )\n        factors.append(uses_aux_hidden_states)',
         '            "dspark",\n        ) or (self.registered_method() is not None\n'
         '              and self.registered_method().uses_aux_hidden_states)\n'
         '        factors.append(uses_aux_hidden_states)\n'
         '        if self.registered_method() is not None:\n'
         '            factors.append((self.method, normalize_value(self.method_config)))')
    edit(p, '            else:\n                raise ValueError(\n                    "num_speculative_tokens was provided but without speculative model."',
         '            elif self.registered_method() is not None:\n                pass  # The plugin configure hook decides whether it needs a model.\n'
         '            else:\n                raise ValueError(\n                    "num_speculative_tokens was provided but without speculative model."')
    edit(p, '                if self.method in ("eagle", "eagle3", "dflash", "dspark"):',
         '                if (self.method in ("eagle", "eagle3", "dflash", "dspark")\n'
         '                        or self.registered_method() is not None):')
    edit(p, '                if self.num_speculative_tokens is None:\n                    raise ValueError(\n                        "A speculative model was provided, but "',
         '                registered = self.registered_method()\n'
         '                if self.num_speculative_tokens is None and (\n'
         '                    registered is None or registered.configure is None\n'
         '                ):\n                    raise ValueError(\n                        "A speculative model was provided, but "')
    edit(p, '            raise ValueError("Adaptive verification only supported with DSpark")\n\n        return self',
         '            raise ValueError("Adaptive verification only supported with DSpark")\n\n'
         '        registered = self.registered_method()\n'
         '        if registered is not None and registered.configure is not None:\n'
         '            registered.configure(self)\n\n        return self')
    edit(p, '    @field_validator("attention_backend", mode="before")',
         upstream(p, "SpeculativeConfig._validate_method_name") + '\n\n    @field_validator("attention_backend", mode="before")')
    edit(p, '    def use_eagle(self) -> bool:',
         upstream(p, "SpeculativeConfig.registered_method") + '\n\n' +
         upstream(p, "SpeculativeConfig.supports_async_scheduling") + '\n\n    def use_eagle(self) -> bool:')
    edit(p, '        return self.method in ("eagle", "eagle3", "mtp", "dflash", "dspark")',
         '        if self.method in ("eagle", "eagle3", "mtp", "dflash", "dspark"):\n'
         '            return True\n        registered = self.registered_method()\n'
         '        return registered is not None and registered.uses_target_hidden_states')

    p = "config/vllm.py"
    edit(p, '        if speculative_config.use_eagle() or speculative_config.uses_draft_model():',
         '        registered = speculative_config.registered_method()\n'
         '        if registered is not None and registered.num_lookahead_tokens is not None:\n'
         '            return registered.num_lookahead_tokens(speculative_config)\n'
         '        if speculative_config.use_eagle() or speculative_config.uses_draft_model():')
    edit(p, '        use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER',
         '        spec = self.speculative_config\n'
         '        if spec is not None and spec.registered_method() is not None:\n'
         '            if envs.VLLM_USE_V2_MODEL_RUNNER is False:\n'
         '                raise ValueError(f"Registered method {spec.method!r} requires Model Runner V2")\n'
         '            return True\n        use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER')
    edit(p, '                    and self.speculative_config.method != "dspark"\n',
         '                    and self.speculative_config.method != "dspark"\n'
         '                    and not self.speculative_config.supports_async_scheduling()\n')
    edit(p, '                and self.speculative_config.method != "dspark"\n            ):',
         '                and self.speculative_config.method != "dspark"\n'
         '                and not self.speculative_config.supports_async_scheduling()\n            ):')
    edit(p, '            ):\n                unsupported.append(f"speculative method \'{speculative_config.method}\'")',
         '            ) and speculative_config.registered_method() is None:\n'
         '                unsupported.append(f"speculative method \'{speculative_config.method}\'")')
    # Unlike the new fork, frozen PP does not relay auxiliary states or tree layout.
    edit(p, '        if speculative_config is not None:\n            # TODO: ngram',
         '        if speculative_config is not None:\n'
         '            if (speculative_config.registered_method() is not None\n'
         '                    and (self.parallel_config.pipeline_parallel_size > 1\n'
         '                         or self.parallel_config.prefill_context_parallel_size > 1\n'
         '                         or self.parallel_config.decode_context_parallel_size > 1)):\n'
         '                unsupported.append("plugin speculative decoding with pipeline/context parallelism")\n'
         '            # TODO: ngram')
    p = "v1/worker/gpu/spec_decode/rejection_sampler.py"
    edit(p, '        expanded_local_pos: torch.Tensor,\n    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:',
         '        expanded_local_pos: torch.Tensor,\n'
         '        seq_lens_upper_bound_np: np.ndarray,\n'
         '        verify_draft_sampled: torch.Tensor,\n'
         '    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:')
    # The native frozen sampler keeps its original processor API. Query bounds
    # are exposed to plugin _verify overrides, not used to alter native logits.
    edit(p, '            processed_logits,\n            draft_logits,\n            draft_sampled,',
         '            processed_logits,\n            draft_logits,\n            verify_draft_sampled,')
    edit(p, '        max_num_logprobs: int,\n    ) -> tuple[torch.Tensor, torch.Tensor, LogprobsTensors | None]:',
         '        max_num_logprobs: int,\n'
         '        verify_draft_sampled: torch.Tensor | None = None,\n'
         '    ) -> tuple[torch.Tensor, torch.Tensor, LogprobsTensors | None]:')
    edit(p, '        cu_num_logits_np = input_batch.cu_num_logits_np\n',
         '        if verify_draft_sampled is None:\n'
         '            verify_draft_sampled = draft_sampled\n'
         '        cu_num_logits_np = input_batch.cu_num_logits_np\n')
    edit(p, '                input_batch.expanded_local_pos[lo:hi],\n',
         '                input_batch.expanded_local_pos[lo:hi],\n'
         '                input_batch.seq_lens_cpu_upper_bound.numpy()[start:end],\n'
         '                verify_draft_sampled[lo:hi],\n')
    edit(p, '        draft_logits: torch.Tensor | None = None,\n    ) -> SamplerOutput:',
         '        draft_logits: torch.Tensor | None = None,\n'
         '        invalid_drafts: torch.Tensor | None = None,\n    ) -> SamplerOutput:')
    edit(p, '        pos = input_batch.positions[input_batch.logits_indices]\n',
         '        pos = input_batch.positions[input_batch.logits_indices]\n'
         '        # Only the verifier sees invalid markers; logits processors keep\n'
         '        # the real token ids for prefix-dependent sampling constraints.\n'
         '        verify_draft_sampled = draft_sampled\n'
         '        if invalid_drafts is not None:\n'
         '            verify_draft_sampled = draft_sampled.masked_fill(invalid_drafts, -1)\n')
    edit(p, '            chunk_logit_limit,\n            max_num_logprobs,\n',
         '            chunk_logit_limit,\n            max_num_logprobs,\n'
         '            verify_draft_sampled,\n')


    p = "v1/worker/gpu/spec_decode/__init__.py"
    edit(p, 'from vllm.config import VllmConfig\n',
         'from vllm.config import VllmConfig\nfrom vllm.v1.spec_decode.registry import get_speculative_method\n')
    edit(p, '    assert speculative_config is not None\n',
         '    assert speculative_config is not None\n'
         '    registered = get_speculative_method(speculative_config.method)\n'
         '    if registered is not None:\n'
         '        return registered.resolve_speculator_cls()(vllm_config, device)\n')

    p = "v1/worker/gpu/spec_decode/speculator.py"
    edit(p, 'class BaseSpeculator(ABC):\n',
         'class BaseSpeculator(ABC):\n'
         '    uses_varlen_verify_graphs: bool = False\n'
         '    num_extra_query_per_req: int = 0\n    num_query_per_req: int = 1\n')
    hooks = '\n\n'.join(upstream(p, 'BaseSpeculator.' + name) for name in
                        ('make_rejection_sampler', 'prepare_verify_inputs', 'post_accept'))
    edit(p, '\n\nclass DraftModelSpeculator(BaseSpeculator):', '\n\n' + hooks + '\n\n\nclass DraftModelSpeculator(BaseSpeculator):')

    # The released Harness calls the AR parent for one seed/prefill step, then
    # grows its own tree. Port the per-call control, not the newer AR runtime.
    reference("v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py")
    p = "v1/worker/gpu/spec_decode/autoregressive/speculator.py"
    edit(p, '        is_profile: bool = False,\n    ) -> torch.Tensor:',
         '        is_profile: bool = False,\n'
         '        num_speculative_tokens: int | None = None,\n    ) -> torch.Tensor:\n'
         '        if num_speculative_tokens is None:\n'
         '            num_speculative_tokens = self.num_speculative_steps\n'
         '        if (not isinstance(num_speculative_tokens, int)\n'
         '                or isinstance(num_speculative_tokens, bool)\n'
         '                or not 0 <= num_speculative_tokens <= self.num_speculative_steps):\n'
         '            raise ValueError("num_speculative_tokens must be an integer between 0 and the configured maximum")')
    edit(p, '            max_seq_len + self.num_speculative_steps, self.max_model_len',
         '            max_seq_len + num_speculative_tokens, self.max_model_len')
    edit(p, '            assert self.method == "eagle3"',
         '            registered = self.speculative_config.registered_method()\n'
         '            assert self.method == "eagle3" or (\n'
         '                registered is not None and registered.uses_aux_hidden_states\n'
         '            ), "Auxiliary hidden states require EAGLE3 or a registered auxiliary-state consumer"')
    edit(p, '        if self.num_speculative_steps == 1:\n'
         '            # Early exit.\n'
         '            return self.draft_tokens[:num_reqs, :1]',
         '        if num_speculative_tokens < self.num_speculative_steps:\n'
         '            # Do not expose stale proposals from a previous longer call.\n'
         '            self.draft_tokens[:num_reqs, num_speculative_tokens:].fill_(-1)\n'
         '        if num_speculative_tokens <= 1:\n'
         '            # Even a zero-token proposal prefills; one seeds Harness tree growth.\n'
         '            return self.draft_tokens[:num_reqs]')
    edit(p, '            uniform_token_count=1,\n'
         '            dp_size=self.dp_size,\n'
         '            dp_rank=self.dp_rank,\n'
         '            need_eager=is_profile,',
         '            uniform_token_count=1,\n'
         '            dp_size=self.dp_size,\n'
         '            dp_rank=self.dp_rank,\n'
         '            # Frozen decode graphs bake in the configured count (also\n'
         '            # the final-step input update). Partial calls cannot replay them.\n'
         '            need_eager=is_profile or num_speculative_tokens != self.num_speculative_steps,')
    edit(p, '            input_batch.seq_lens_cpu_upper_bound,\n        )',
         '            input_batch.seq_lens_cpu_upper_bound,\n'
         '            num_speculative_tokens,\n        )')
    edit(p, '        seq_lens_cpu_upper_bound: torch.Tensor,\n    ) -> None:',
         '        seq_lens_cpu_upper_bound: torch.Tensor,\n'
         '        num_speculative_steps: int | None = None,\n    ) -> None:', count=2)
    edit(p, '        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,\n    ) -> None:',
         '        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,\n'
         '        num_speculative_steps: int | None = None,\n    ) -> None:', count=2)
    # Defaults retain the existing capture callback ABI and configured graphs.
    edit(p, '        num_speculative_steps: int | None = None,\n    ) -> None:\n',
         '        num_speculative_steps: int | None = None,\n    ) -> None:\n'
         '        if num_speculative_steps is None:\n'
         '            num_speculative_steps = self.num_speculative_steps\n', count=4)
    edit(p, '        for step in range(1, self.num_speculative_steps):',
         '        for step in range(1, num_speculative_steps):', count=2)
    edit(p, '                step < self.num_speculative_steps - 1',
         '                step < num_speculative_steps - 1')
    edit(p, '                    cudagraph_runtime_mode=batch_desc.cg_mode,\n                )',
         '                    cudagraph_runtime_mode=batch_desc.cg_mode,\n'
         '                    num_speculative_steps=num_speculative_steps,\n                )')
    edit(p, '            batch_desc.cg_mode,\n        )',
         '            batch_desc.cg_mode,\n            num_speculative_steps,\n        )')
    edit(p, '                cudagraph_runtime_mode,\n            )',
         '                cudagraph_runtime_mode,\n                num_speculative_steps,\n            )')
    edit(p, '            self.max_model_len,\n            self.num_speculative_steps,',
         '            self.max_model_len,\n            num_speculative_steps,')

    p = "v1/attention/backend.py"
    edit(p, '    def _init_reorder_batch_threshold(',
         upstream(p, 'AttentionMetadataBuilder.get_varlen_cudagraph_max_query_len')
         + '\n\n    def _init_reorder_batch_threshold(')
    p = "v1/worker/gpu/attn_utils.py"
    edit(p, '\ndef init_kv_cache(', '\n' + upstream(p, 'get_varlen_cudagraph_unsupported_backend') + '\n\ndef init_kv_cache(')

    p = "v1/worker/gpu/model_runner.py"
    edit(p, 'from vllm.v1.worker.gpu.attn_utils import (',
         'from vllm.v1.attention.backend import AttentionCGSupport\n'
         'from vllm.v1.worker.gpu.attn_utils import (\n    get_varlen_cudagraph_unsupported_backend,')
    edit(p, '            if self.speculative_config.method in ("eagle3", "dflash", "dspark"):',
         '            registered = self.speculative_config.registered_method()\n'
         '            if (self.speculative_config.method in ("eagle3", "dflash", "dspark")\n'
         '                    or (registered is not None and registered.uses_aux_hidden_states)):')
    edit(p, '                self.rejection_sampler = RejectionSampler(\n'
         '                    self.sampler,\n                    self.speculative_config,\n'
         '                    self.device,\n                )',
         '                self.rejection_sampler = self._build_rejection_sampler()')
    edit(p, '    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:',
         upstream(p, 'GPUModelRunner._build_rejection_sampler').replace(
             '            watermark_key=speculative_target_watermark_key(\n'
             '                self.vllm_config.watermark_config\n            ),\n', '') + '\n\n'
         '    def _use_varlen_verify_graphs(self, attn_cg_support) -> bool:\n'
         '        if self.adaptive_verification is not None:\n            return True\n'
         '        if self.speculator is None or not self.speculator.uses_varlen_verify_graphs:\n            return False\n'
         '        target_names = {name for group in self.kv_cache_config.kv_cache_groups\n'
         '                        for name in group.layer_names}\n'
         '        if isinstance(self.speculator, DraftModelSpeculator):\n'
         '            target_names -= self.speculator.draft_attn_layer_names\n'
         '        unsupported = get_varlen_cudagraph_unsupported_backend(\n'
         '            self.attn_groups, self.vllm_config, self.decode_query_len, target_names)\n'
         '        support, backend = self.model_state.get_additional_cg_support()\n'
         '        if unsupported is not None or support != AttentionCGSupport.ALWAYS:\n'
         '            raise ValueError(f"Variable-length verify graphs unsupported by {unsupported or backend}")\n'
         '        return True\n\n'
         '    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:')
    edit(p, '        if self.adaptive_verification is not None:\n            self.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE',
         '        varlen_verify_graphs = self._use_varlen_verify_graphs(attn_cg_support)\n'
         '        if self.adaptive_verification is not None:\n'
         '            self.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE\n'
         '        elif varlen_verify_graphs and self.compilation_config.cudagraph_mode.has_full_cudagraphs():\n'
         '            if self.compilation_config.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:\n'
         '                self.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE')
    edit(p, '            varlen_decode=self.adaptive_verification is not None,', '            varlen_decode=varlen_verify_graphs,')
    edit(p, '            assert num_tokens % self.decode_query_len == 0\n',
         '            assert num_tokens % self.decode_query_len == 0\n'
         '        elif self.speculator is not None and self.speculator.num_extra_query_per_req:\n'
         '            num_reqs = min(num_reqs, self.max_num_tokens // self.speculator.num_query_per_req)\n'
         '            num_tokens = min(num_tokens, self.max_num_tokens\n'
         '                             - num_reqs * self.speculator.num_extra_query_per_req)\n')
    # Release calibration opts into real state work. The frozen runner swallows
    # this flag in **kwargs; its default warmup/capture paths must stay unchanged.
    edit(p, '        is_profile: bool = False,\n        **kwargs,',
         '        is_profile: bool = False,\n'
         '        valid_dummy_state_slots: bool = False,\n        **kwargs,')
    edit(p, '                context_len=context_len,\n',
         '                context_len=context_len,\n'
         '                valid_dummy_state_slots=valid_dummy_state_slots,\n')
    edit(p, '        context_len: int = 0,\n    ) -> ModelRunnerOutput | IntermediateTensors | None:',
         '        context_len: int = 0,\n'
         '        valid_dummy_state_slots: bool = False,\n'
         '    ) -> ModelRunnerOutput | IntermediateTensors | None:')
    edit(p, '                block_tables, slot_mappings = self.prepare_dummy_attn(input_batch)\n'
         '                if context_len:\n'
         '                    set_dummy_context(\n'
         '                        input_batch,\n'
         '                        self.block_tables,\n'
         '                        context_len,\n'
         '                        self.kv_cache_config.num_blocks,\n'
         '                        self.max_model_len,\n'
         '                    )',
         '                block_tables, slot_mappings = self.prepare_dummy_attn(\n'
         '                    input_batch, valid_dummy_state_slots, context_len=context_len\n'
         '                )')
    edit(p, symbol(changed[p], 'GPUModelRunner.prepare_dummy_attn'), '''    def prepare_dummy_attn(
        self, input_batch: InputBatch, valid_state_slots: bool = False,
        *, context_len: int = 0,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        block_tables = self.block_tables.get_dummy_block_tables(input_batch.num_reqs)
        slot_mappings = pcp.maybe_get_pcp_dummy_slot_mappings(
            self.pcp_manager, self.block_tables, input_batch.num_tokens
        )
        if context_len:
            set_dummy_context(
                input_batch, self.block_tables, context_len,
                self.kv_cache_config.num_blocks, self.max_model_len,
            )
        if valid_state_slots:
            # set_dummy_context rewrites the tables, including reserved block 0.
            # Assign real, disjoint recurrent-state rows AFTER that rewrite. Keep
            # attention context tables/slot mappings and inactive rows unchanged.
            for table, group in zip(block_tables, self.kv_cache_config.kv_cache_groups):
                spec = group.kv_cache_spec
                if not isinstance(spec, MambaSpec):
                    continue
                width = 1 + spec.num_speculative_blocks
                count = input_batch.num_reqs * width
                if count >= self.kv_cache_config.num_blocks or width > table.shape[1]:
                    raise ValueError("Insufficient distinct Mamba state slots for calibration")
                slots = torch.arange(1, count + 1, dtype=table.dtype, device=table.device)
                slots = slots.view(input_batch.num_reqs, width)
                table.zero_()
                if self.cache_config.mamba_cache_mode == "align":
                    starts = ((input_batch.seq_lens - 1) // spec.block_size).clamp(min=0)
                    columns = starts[:, None] + torch.arange(width, device=table.device)
                    table.scatter_(1, columns.to(torch.int64), slots)
                else:
                    table[:, :width].copy_(slots)
            # Frozen eager metadata otherwise sees q>1 dummy rows as prefill:
            # make the calibration decode shape reach the real GDN verifier.
            if self.speculative_config is not None:
                drafts = input_batch.num_scheduled_tokens - 1
                if np.all(drafts <= self.speculative_config.num_speculative_tokens):
                    input_batch.num_draft_tokens_per_req = drafts
                    input_batch.num_draft_tokens = int(drafts.sum())
        return block_tables, slot_mappings''')
    edit(p, '            **self.model_state.prepare_inputs(input_batch, self.req_states),\n        }',
         '            **self.model_state.prepare_inputs(input_batch, self.req_states),\n        }\n'
         '        if self.speculator is not None:\n'
         '            # After slot/attention metadata and model positions, before target forward.\n'
         '            self.speculator.prepare_verify_inputs(input_batch, model_inputs, dummy_run=dummy_run)')
    edit(p, '        sampler_output, num_sampled, num_rejected = self.sample(\n'
         '            hidden_states, input_batch, grammar_output\n        )',
         '        sampler_output, num_sampled, num_rejected = self.sample(\n'
         '            hidden_states, input_batch, grammar_output\n        )\n'
         '        if self.speculator is not None:\n'
         '            # Commit the accepted parent path before request-state advance/propose.\n'
         '            self.speculator.post_accept(input_batch, sampler_output, num_sampled, num_rejected)')

    p = "v1/attention/backends/gdn_attn.py"
    edit(p, 'from vllm.config import VllmConfig\n',
         'from vllm.config import VllmConfig\n'
         'from vllm.model_executor.layers.mamba.gdn.verify_extension import (\n'
         '    LinearAttnVerifyExtension, create_linear_attn_verify_extension,\n)\n')
    edit(p, '    token_chunk_offset_ptr: torch.Tensor | None = None\n',
         '    token_chunk_offset_ptr: torch.Tensor | None = None\n'
         '    verify_extension: LinearAttnVerifyExtension | None = None\n')
    edit(p, '        self._init_reorder_batch_threshold(1, self.use_spec_decode)\n',
         '        self._init_reorder_batch_threshold(1, self.use_spec_decode)\n'
         '        self.verify_extension = (create_linear_attn_verify_extension(vllm_config)\n'
         '                                 if self.use_spec_decode else None)\n')
    edit(p, '            token_chunk_offset_ptr=token_chunk_offset_ptr,\n',
         '            token_chunk_offset_ptr=token_chunk_offset_ptr,\n'
         '            verify_extension=self.verify_extension,\n')

    p = "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
    edit(p, 'from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention\n',
         'from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention\n'
         'from vllm.model_executor.layers.mamba.gdn.verify_extension import (\n'
         '    GDNSpecRows, LinearAttnVerifyExtension,\n)\n')
    edit(p, '    def _forward_core(\n', upstream(p, 'QwenGatedDeltaNetAttention._forward_spec_rows') + '\n\n    def _forward_core(\n')
    # Restrict the following hunks to the common core; keep native fast paths unchanged.
    core = symbol(changed[p], 'QwenGatedDeltaNetAttention._forward_core')
    newcore = core.replace('        # 1. Convolution sequence transformation',
                          '        verify_extension = attn_metadata.verify_extension\n'
                          '        # 1. Convolution sequence transformation')
    conv_start = newcore.index('            mixed_qkv_spec = causal_conv1d_update(')
    conv_end = newcore.index('\n\n        # 1.2:', conv_start)
    native_conv = newcore[conv_start:conv_end]
    newcore = newcore[:conv_start] + (
        '            if verify_extension is not None:\n'
        '                extension_out_spec = mixed_qkv_spec.new_empty(\n'
        '                    mixed_qkv_spec.shape[0], self.num_v_heads // self.tp_size, self.head_v_dim)\n'
        '                self._forward_spec_rows(verify_extension, attn_metadata, mixed_qkv_spec,\n'
        '                                        a_spec, b_spec, conv_state, conv_weights,\n'
        '                                        ssm_state, extension_out_spec)\n'
        '                mixed_qkv_spec = None\n'
        '            else:\n' + ''.join('    ' + line + '\n' for line in native_conv.splitlines()).rstrip('\n')
    ) + newcore[conv_end:]
    newcore = newcore.replace('        # 2.1: Process the multi-query part\n        if spec_sequence_masks is not None:',
                             '        # 2.1: Process the multi-query part\n'
                             '        if spec_sequence_masks is not None and verify_extension is not None:\n'
                             '            core_attn_out_spec = extension_out_spec.unsqueeze(0)\n'
                             '            last_recurrent_state = None\n'
                             '        elif spec_sequence_masks is not None:')
    edit(p, core, newcore)
    method = symbol(changed[p], 'QwenGatedDeltaNetAttention._can_use_fused_gdn_mtp_decode')
    edit(p, method, method.replace('            attn_metadata.spec_sequence_masks is not None',
                                  '            attn_metadata.verify_extension is None\n'
                                  '            and attn_metadata.spec_sequence_masks is not None'))

    p = "_xpu_ops.py"
    edit(p, '    assert isinstance(attn_metadata, GDNAttentionMetadata)\n',
         '    assert isinstance(attn_metadata, GDNAttentionMetadata)\n'
         '    if (attn_metadata.verify_extension is not None\n'
         '            and attn_metadata.spec_sequence_masks is not None):\n'
         '        from vllm.model_executor.layers.mamba.gdn.verify_extension_xpu import forward_xpu_spec_rows\n'
         '        forward_xpu_spec_rows(self, attn_metadata, projected_states_qkvz,\n'
         '                              projected_states_ba, core_attn_out, z)\n'
         '        return\n')
    p = "platforms/xpu.py"
    edit(p, '        # TurboQuant KV cache: route directly to TQ backend',
         '        if selected_backend == AttentionBackendEnum.CUSTOM:\n'
         '            # Explicit plugin backend, never replace its tree mask with causal attention.\n'
         '            # get_path raises when CUSTOM has not been registered.\n'
         '            return selected_backend.get_path()\n'
         '        # TurboQuant KV cache: route directly to TQ backend')

    p = "model_executor/layers/mamba/gdn/verify_extension_xpu.py"
    originals[p] = ""
    changed[p] = (HERE / "verify_extension_xpu.py").read_text()

    # Opaque device addresses may have bit 63 set on XPU. Preserve their
    # bits through PyTorch's signed-int64 scalar-assignment boundary.
    p = "v1/worker/mamba_utils.py"
    edit(p, '                    self.state_base_addrs[idx] = state.data_ptr()',
         '                    address = state.data_ptr()\n'
         '                    self.state_base_addrs[idx] = address if address < (1 << 63) else address - (1 << 64)')
    edit(p, '            self.block_table_ptrs[i] = bt.data_ptr()',
         '            address = bt.data_ptr()\n'
         '            self.block_table_ptrs[i] = address if address < (1 << 63) else address - (1 << 64)')

    # Guard unchanged dependencies that provide the scheduler/stock-XPU lifecycle.
    for p in ("version.py", "v1/attention/backends/registry.py", "v1/core/sched/scheduler.py",
              "v1/worker/gpu/kv_connector.py", "v1/worker/gpu/warmup.py",
              "v1/worker/xpu_model_runner.py", "v1/worker/xpu_worker.py",
              "v1/worker/gpu/block_table.py", "v1/worker/gpu/input_batch.py",
              "v1/worker/gpu/model_states/mamba_hybrid.py", "v1/attention/backends/utils.py",
              "v1/worker/gpu/spec_decode/autoregressive/cudagraph_utils.py",
              "v1/worker/gpu/cudagraph_utils.py", "v1/worker/gpu/dp_utils.py"):
        load(p)
    hunks, files = [], []
    for p in sorted(changed):
        before, after = originals[p], changed[p]
        compile(after, p, "exec")
        if before != after:
            hunks.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                            fromfile="a/vllm/" + p, tofile="b/vllm/" + p))
        files.append(dict(path=p, before_sha256=digest(before) if before else None,
                          after_sha256=digest(after)))
    patch = "".join(hunks)
    (HERE / "engine.patch").write_text(patch)
    (HERE / "manifest.json").write_text(json.dumps(dict(
        base_version=VERSION, base_image=IMAGE, hook_reference=OFFICIAL,
        patch_sha256=digest(patch), files=files), indent=2) + "\n")
    print(f"Built {sum(originals[p] != changed[p] for p in changed)} patched files; {len(files)} pinned files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="immutable exported vllm package")
    parser.add_argument("--official", type=Path, required=True, help="official reference's vllm directory")
    args = parser.parse_args()
    build(args.base.resolve(), args.official.resolve())
