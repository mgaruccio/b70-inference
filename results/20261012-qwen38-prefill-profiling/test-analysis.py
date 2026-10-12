import json
from pathlib import Path
import runpy
import tempfile
import unittest

ANALYZE = runpy.run_path(str(Path(__file__).with_name('analyze.py')), run_name='analysis_test')['analyze']


class Tests(unittest.TestCase):
    def test_gpu_durations_and_cpu_identity_not_gpu_start_assign_phase(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / 'torch').mkdir()
            (p / 'summary.json').write_text(json.dumps({'budget':8192,'prefix_cache':'on','length':2,
                'unprofiled':[{'elapsed_s':1},{'elapsed_s':2},{'elapsed_s':3}], 'profiled':{'elapsed_s':2.1}}))
            (p / 'iterations.json').write_text(json.dumps([{'context_tokens':1},{'context_tokens':1}]))
            events = [
                {'cat':'user_annotation','name':'execute_context_1(1)_generation_0(0)','ts':100,'dur':50},
                {'cat':'user_annotation','name':'execute_context_1(1)_generation_0(0)','ts':200,'dur':50},
                {'cat':'cpu_op','name':'_xpu_C::int4_gemm_w4a16','ts':110,'dur':999999,'args':{'External id':1}},
                {'cat':'cpu_op','name':'aten::mm','ts':210,'dur':999999,'args':{'External id':2}},
                {'cat':'xpu_runtime','name':'launch','ts':211,'dur':1,'args':{'correlation':22,'External id':2}},
                {'cat':'kernel','ph':'X','name':'gemm_kernel','ts':10000,'dur':2000,'args':{'External id':1}},
                {'cat':'kernel','ph':'X','name':'gemm_kernel','ts':12000,'dur':3000,'args':{'correlation':22}},
            ]
            (p / 'torch/rank0.pt.trace.json').write_text(json.dumps({'traceEvents':events}))
            result = ANALYZE(p)
            self.assertEqual(result['device_groups']['INT4 GEMM']['device_ms'],2)
            self.assertEqual(result['device_groups']['dense GEMM']['device_ms'],3)
            self.assertEqual(result['phases'][0]['groups']['INT4 GEMM']['calls'],1)
            self.assertEqual(result['phases'][1]['groups']['dense GEMM']['calls'],1)
            self.assertEqual(result['unmatched_phase_kernel_count'],0)
            self.assertAlmostEqual(result['profile_overhead_pct'],5)


if __name__ == '__main__':
    unittest.main()
