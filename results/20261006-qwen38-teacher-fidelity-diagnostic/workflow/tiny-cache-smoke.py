import sys, torch, json
sys.path.insert(0,'/home/mike/code/b70-inference/scripts/experiments')
import qwen38_teacher_fidelity as diag
import qwen38_train_mtp as trainer
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
cfg=Qwen3_5TextConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,head_dim=8,linear_num_key_heads=4,linear_num_value_heads=4,linear_key_head_dim=8,linear_value_head_dim=8,linear_conv_kernel_dim=4,layer_types=['linear_attention']*3+['full_attention'],rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':0.25,'mrope_section':[1,1,1]})
cfg._attn_implementation='sdpa'
torch.manual_seed(42)
text=Qwen3_5TextModel(cfg).to(torch.bfloat16).eval().requires_grad_(False)
head=torch.nn.Linear(32,64,bias=False).to(torch.bfloat16).eval().requires_grad_(False)
t=object.__new__(trainer.FrozenTarget);t.text=text;t.model=text;t.head=head;t.device=torch.device('cpu')
report=diag.run_hf_prefix(t,[1,2,3,4,5,6,7,8,9])
print('REAL_TINY_HYBRID_CACHE_SMOKE',json.dumps({k:v for k,v in report.items() if k not in ('prefix','full_rows','cached_rows','past')},default=str)); print('CACHE_FACTS',diag.cache_facts(report['past']))
assert report['errors']['pass']
assert not any(p.grad is not None for p in text.parameters())
print('SMOKE_PASS_NOT_NATIVE_PARITY')
