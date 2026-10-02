import json, os, platform, time
from pathlib import Path
import torch, transformers
from transformers import AutoModelForImageTextToText, AutoTokenizer

root = Path.home()
model_path = root / 'glimmer-model'
outdir = root / 'mtp-pilot-run'
outdir.mkdir(exist_ok=True)
torch.manual_seed(17)
t0 = time.perf_counter()
model = AutoModelForImageTextToText.from_pretrained(str(model_path), dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa').eval()
model.requires_grad_(False)
tokenizer = AutoTokenizer.from_pretrained(str(model_path))
ids = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Write a Python function that adds two integers.'}], reasoning_strength='low', add_generation_prompt=True, return_tensors='pt', return_dict=True)['input_ids'].to('cuda')
assert ids.shape[1] < 512
with torch.inference_mode():
    pre = model(input_ids=ids, use_cache=True, output_hidden_states=True)
    h = pre.hidden_states[-1][:, -1]
    direct = model.get_output_embeddings()(h)
    anchor = pre.logits[:, -1].argmax(-1).reshape(1, 1)
    assert torch.equal(direct.argmax(-1), anchor.flatten())
    one = model(input_ids=anchor, past_key_values=pre.past_key_values, use_cache=True, output_hidden_states=True)
    second = one.logits[:, -1].argmax(-1).reshape(1, 1)
    two = model(input_ids=second, past_key_values=one.past_key_values, use_cache=True, output_hidden_states=True)
    next_token = two.logits[:, -1].argmax(-1)
    pre2 = model(input_ids=ids, use_cache=True, output_hidden_states=True)
    block = model(input_ids=torch.cat([anchor, second], 1), past_key_values=pre2.past_key_values, use_cache=True, output_hidden_states=True)
    assert torch.equal(block.logits[:, 0].argmax(-1), second.flatten())
    assert torch.equal(block.logits[:, 1].argmax(-1), next_token)
    before = block.past_key_values.get_seq_length()
    block.past_key_values.crop(-1)
    assert block.past_key_values.get_seq_length() == ids.shape[1] + 1
    retry = model(input_ids=second, past_key_values=block.past_key_values, use_cache=True, output_hidden_states=True)
    assert torch.equal(retry.logits[:, -1].argmax(-1), next_token)
    result = {'model_revision': 'a4e59da52a7bc87ae7251dd5545c0dd437c44b68', 'torch': torch.__version__, 'transformers': transformers.__version__, 'python': platform.python_version(), 'gpu': torch.cuda.get_device_name(), 'model_class': type(model).__name__, 'hidden_shape': list(h.shape), 'hidden_finite': bool(h.isfinite().all()), 'input_embedding_class': type(model.get_input_embeddings()).__name__, 'output_embedding_class': type(model.get_output_embeddings()).__name__, 'prompt_tokens': ids.shape[1], 'anchor': int(anchor.item()), 'draft_token': int(second.item()), 'next_token': int(next_token.item()), 'cache_class': type(block.past_key_values).__name__, 'cache_length_before_crop': before, 'cache_crop_and_replay_ok': True, 'batched_verification_matches_single_steps': True, 'peak_cuda_allocated_gb': torch.cuda.max_memory_allocated()/1e9, 'elapsed_s_including_load': time.perf_counter()-t0}
    print(json.dumps(result, indent=2), flush=True)
    (outdir / 'model-canary.json').write_text(json.dumps(result, indent=2)+'\n')
