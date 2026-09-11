"""CPU-only smoke, executed inside the pinned serving image (not Pi's runtime)."""
import importlib.util
from types import SimpleNamespace
import torch

spec = importlib.util.spec_from_file_location('annotations', '/experiment/target-annotations.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
root = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
shared = root[0]
x = torch.ones(1, 4)
expected = root(x).detach().clone()

class Worker:
    model_runner = SimpleNamespace(model=root)
    def profile(self, is_start=True, profile_prefix=None):
        return is_start

mod.install_worker_profile(Worker)
w = Worker()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
    assert w.profile(True)
    actual = root(x)
    shared(x)  # Must not be attributed to target outside target root forward.
    assert not w.profile(False)
    root(x)  # Hooks must have been removed.
assert torch.equal(expected, actual)
names = [event.name for event in prof.events() if event.name.startswith('b70_target/')]
assert len(names) == 3, names
assert not root._forward_pre_hooks and not shared._forward_pre_hooks
assert not root._forward_hooks and not shared._forward_hooks
print({'status': 'passed', 'torch': torch.__version__, 'annotations': names,
       'output_equal': True, 'shared_module_excluded': True, 'hooks_removed': True})
