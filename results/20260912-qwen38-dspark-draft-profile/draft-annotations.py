"""Disposable eager DSpark annotations, active only in the native profile window."""
import functools
import json

import torch


def install():
    pass


def install_worker_profile(worker_class):
    original = worker_class.profile
    handles, restorations = [], []
    stacks = {}
    state = {'depth': 0, 'metadata_logged': False}

    def describe(value):
        if isinstance(value, torch.Tensor):
            return {'shape': list(value.shape), 'dtype': str(value.dtype), 'device': str(value.device)}
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        return type(value).__name__

    def remove():
        for handle in handles:
            handle.remove()
        handles.clear()
        for owner, name, existed, previous in reversed(restorations):
            if existed:
                setattr(owner, name, previous)
            else:
                delattr(owner, name)
        restorations.clear()
        stacks.clear()
        state.update(depth=0, metadata_logged=False)

    def wrap(owner, name, label, metadata=False):
        original_method = getattr(owner, name)
        restorations.append((owner, name, name in owner.__dict__, owner.__dict__.get(name)))

        @functools.wraps(original_method)
        def call(*args, **kwargs):
            if metadata and not state['metadata_logged']:
                attn = args[1] if len(args) > 1 else kwargs.get('attn_metadata')
                fields = ('causal', 'max_query_len', 'max_seq_len', 'num_actual_tokens',
                          'query_start_loc', 'seq_lens', 'block_table')
                summary = {key: {field: describe(getattr(value, field)) for field in fields if hasattr(value, field)}
                           for key, value in (attn or {}).items()}
                print('B70_DRAFT_METADATA: ' + json.dumps(summary), flush=True)
                state['metadata_logged'] = True
            with torch.profiler.record_function('b70_draft/phase:' + label):
                state['depth'] += 1
                try:
                    return original_method(*args, **kwargs)
                finally:
                    state['depth'] -= 1
        setattr(owner, name, call)

    def attach(worker):
        if handles:
            return
        speculator = worker.model_runner.speculator
        if type(speculator).__name__ != 'DSparkSpeculator':
            raise RuntimeError('Expected corrected DSparkSpeculator')
        root = speculator.model
        for name, module in root.named_modules():
            label = f'b70_draft/{name or "root"}:{type(module).__name__}'

            def before(mod, args, label=label):
                if state['depth']:
                    context = torch.profiler.record_function(label)
                    context.__enter__()
                    stacks.setdefault(id(mod), []).append(context)

            def after(mod, args, output):
                stack = stacks.get(id(mod))
                if stack:
                    stack.pop().__exit__(None, None, None)

            handles.append(module.register_forward_pre_hook(before))
            handles.append(module.register_forward_hook(after, always_call=True))
            fields = ('causal', 'sliding_window', 'num_heads', 'num_kv_heads', 'head_dim', 'kv_cache_dtype')
            if hasattr(module, 'causal') or hasattr(module, 'kv_cache_dtype'):
                detail = {field: describe(getattr(module, field)) for field in fields if hasattr(module, field)}
                detail['impl'] = type(getattr(module, 'impl', None)).__name__
                print('B70_DRAFT_MODULE: ' + json.dumps({'name': name, 'type': type(module).__name__, **detail}), flush=True)
        wrap(speculator, '_generate_draft', 'generate')
        wrap(speculator, '_run_model', 'backbone', metadata=True)
        wrap(speculator, '_sample_sequential', 'sampling')
        wrap(root, 'precompute_and_store_context_kv', 'context_kv')
        print(f'B70_DRAFT_ANNOTATIONS: {type(root).__name__}, {len(handles)//2} modules', flush=True)

    @functools.wraps(original)
    def profile(worker, is_start=True, profile_prefix=None):
        if is_start:
            result = original(worker, is_start=True, profile_prefix=profile_prefix)
            try:
                attach(worker)
            except BaseException:
                remove()
                raise
            return result
        try:
            return original(worker, is_start=False, profile_prefix=profile_prefix)
        finally:
            remove()

    worker_class.profile = profile
