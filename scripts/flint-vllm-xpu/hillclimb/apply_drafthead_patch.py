"""Use the Flint XPU int8 draft lm_head for DFlash 2 candidate selection (drafter only)."""
import pathlib, sys
p = pathlib.Path(sys.argv[1]); s = p.read_text()
old = ("        return self.candidate_logits_processor.get_top_k_tokens(\n"
       "            self.lm_head, hidden_states, self.model.candidate_selector.top_k\n"
       "        )\n")
new = ("        from vllm.models.flint.ops.xpu_draft_head import get_draft_head\n"
       "\n"
       "        return self.candidate_logits_processor.get_top_k_tokens(\n"
       "            get_draft_head(self, self.lm_head), hidden_states,\n"
       "            self.model.candidate_selector.top_k,\n"
       "        )\n")
assert s.count(old) == 1
p.write_text(s.replace(old, new))
print('drafthead patch applied')
