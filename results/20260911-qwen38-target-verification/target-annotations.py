"""Temporary eager-only target forward annotations; reused worker-import shim."""
import functools
import torch


def install():
    # The existing disposable worker patch calls this at import time.
    pass


def install_worker_profile(worker_class):
    original = worker_class.profile
    handles = []
    stacks = {}
    state = {'depth': 0}

    def remove():
        for handle in handles:
            handle.remove()
        handles.clear()
        stacks.clear()
        state['depth'] = 0

    def attach(root):
        if handles:
            return
        for name, module in root.named_modules():
            label = f'b70_target/{name or "root"}:{type(module).__name__}'

            def before(mod, args, label=label):
                if mod is root:
                    state['depth'] += 1
                if state['depth']:
                    context = torch.profiler.record_function(label)
                    context.__enter__()
                    stacks.setdefault(id(mod), []).append(context)

            def after(mod, args, output):
                stack = stacks.get(id(mod))
                if stack:
                    stack.pop().__exit__(None, None, None)
                if mod is root:
                    state['depth'] -= 1

            handles.append(module.register_forward_pre_hook(before))
            handles.append(module.register_forward_hook(after, always_call=True))
        print(f'B70_TARGET_ANNOTATIONS: {type(root).__name__}, {len(handles)//2} modules', flush=True)

    @functools.wraps(original)
    def profile(worker, is_start=True, profile_prefix=None):
        if is_start:
            result = original(worker, is_start=True, profile_prefix=profile_prefix)
            attach(worker.model_runner.model)
            return result
        try:
            return original(worker, is_start=False, profile_prefix=profile_prefix)
        finally:
            remove()

    worker_class.profile = profile
