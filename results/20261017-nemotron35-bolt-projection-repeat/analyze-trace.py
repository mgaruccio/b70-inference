#!/usr/bin/env python3
"""Compare real-server printed boundaries; equal sums never imply bitwise equality."""
import datetime
import json
import math
from pathlib import Path
import re
import runpy

ROOT=Path('/home/mike/b70-evals/20261017-nemotron35-bolt-projection-repeat')


def load(path):
    return json.loads(path.read_text())


def parse_boundaries(text):
    records=[]
    for raw in text.splitlines():
        line=re.sub(r'\x1b\[[0-9;]*m','',raw)
        header=re.search(r'common_debug_cb_eval:\s+(\S+)\s+=.*=\s*\{([\d, ]+)\}',line)
        if header:
            name,shape=header.groups()
            operation=re.search(r'\([^)]*\)\s+([A-Z0-9_]+)\(',line)
            assert operation,line
            op=operation.group(1)
            assert name in ('embd','norm-0','attn_norm-0','mamba2_y_add_d-0','mamba_out-0','nemotron_h_block_out-0','result_output') or op in ('MUL_MAT','SSM_CONV','SSM_SCAN'),name
            records.append(dict(name=name,op=op,header=line[line.index('common_debug_cb_eval:'):],shape=[int(n) for n in shape.split(',')],printed_sums=[]))
        value=re.search(r'\bsum =\s*([^\s]+)',line)
        if value:
            assert records,'sum without tensor header'
            number=float(value.group(1))
            assert math.isfinite(number),'nonfinite printed tensor sum'
            records[-1]['printed_sums'].append(number)
    for record in records:
        assert len(record['shape'])==4 and len(record['printed_sums'])==record['shape'][3],record
    return records


def replay_segments(log):
    starts=list(re.finditer(r'^.*\| task\s+(\d+)\s+\| new prompt,.*task\.n_tokens = (\d+).*$',log,re.MULTILINE))
    assert len(starts)>=6
    chosen=starts[-6:]
    assert all(int(m.group(2))==94 for m in chosen),'Last six model requests are not the declared94-token replays'
    return [(int(m.group(1)),log[m.start():chosen[i+1].start() if i+1<len(chosen) else len(log)]) for i,m in enumerate(chosen)]


def parse_comparisons(text):
    keys=('sample','baseline','weight_equal','input_equal','output_equal','weight_bytes','input_bytes','output_bytes')
    pattern=r'BOLT_PROJECTION_REPEAT sample=(\d+) baseline=([01]) weight_equal=([01]) input_equal=([01]) output_equal=([01]) weight_bytes=(\d+) input_bytes=(\d+) output_bytes=(\d+) weight_type=(\S+)'
    return [dict(zip(keys,map(int,m.groups()[:8])),weight_type=m.group(9)) for m in re.finditer(pattern,text)]


def main():
    c=runpy.run_path(str(ROOT/'run-trace.py'))
    p=runpy.run_path(str(c['PREVIOUS']/'run-controls.py'))
    replay=runpy.run_path(str(c['PREFIX']/'run-replay.py'))
    helper=runpy.run_path(str(p['GROUPED']/'run-grouped-diagnostic.py'))
    runner=runpy.run_path(str(p['OLD']/'run-bolt-cell-ssm-prefill-r1.py'))
    api=runpy.run_path(str(p['OLD']/'probe-api.py'))
    score=runpy.run_path(str(c['PREFIX']/'analyze-replay.py'))['score_row']
    command,binaries=helper['command'],c['binaries']
    assert command('hostname').strip()=='inference-host'
    out=ROOT/'run-r1';summary=load(out/'summary.json')
    assert summary['restored'] and not summary['timing_permitted'] and not summary['promotion_permitted']
    result=dict(matrix_status=summary['status'],matrix_error=summary.get('error'),cells={},timing_permitted=False,promotion_permitted=False,
                caveat='Printed float sums are localization hints only. Equal sums are not bitwise identity; callback readbacks/scheduler splits can mask or alter behavior.')
    prefix=load(out/'frozen-prefix.json')['generated_prefix_ids']
    for label,data in summary['cells'].items():
        cell=out/label;log=(cell/'server.log').read_text()
        item=dict(status=data['status'],tracing=data['tracing'])
        result['cells'][label]=item
        item['native385_passed']='BOLT_SSM_PREFILL_NUMERICAL_PASSED' in log
        if item['native385_passed']:
            rows=[tuple(map(int,x)) for x in re.findall(r'(\d+)/(\d+) tests passed',log.split('BOLT_SSM_PREFILL_NUMERICAL_PASSED',1)[0])]
            assert rows==[(n,n) for n in [56,56,20,20,82,82,34,34,1] for _ in range(2)]
        if (cell/'quality/summary.json').exists():
            q=load(cell/'quality/summary.json')
            item['quality']=dict(passed=sum(bool(x['pass']) for x in q['checks']),total=len(q['checks']),gate=q['pass'])
        if data['status']!='completed':continue
        assert item['native385_passed'] and item['quality']==dict(passed=20,total=20,gate=True)
        model_log=log.split(c['MARKER'],1)[1]
        assert ('BOLT_TENSOR_TRACE_ENABLED' in model_log)==data['tracing']
        assert '[SYCL][OP]' not in model_log and 'GGML_ASSERT' not in model_log and 'SYCL exception' not in model_log
        ids=data['prompt_ids']
        assert len(ids)==55 and load(cell/'tokenize-plain.response.json')['tokens']==load(cell/'tokenize-special.response.json')['tokens']==ids
        expected_prompt=load(out/'frozen-prefix.json')['rendered_prompt']
        assert load(cell/'template.response.json')['prompt']==load(cell/'detokenize.response.json')['content']==expected_prompt
        warm=load(cell/'warmup.result.json')
        assert warm['usage']['prompt_tokens']==55 and warm['usage']['completion_tokens']==128 and warm['timings']['cache_n']==0
        samples=[];boundaries=[]
        segments=replay_segments(log)
        comparisons=parse_comparisons(log)
        item['operand_comparisons']=comparisons
        if data['tracing']:
            assert len(comparisons)==6 and [r['sample'] for r in comparisons]==list(range(6))
            assert [r['baseline'] for r in comparisons]==[1,0,0,0,0,0]
            assert len({(r['weight_bytes'],r['weight_type']) for r in comparisons})==1
            assert all(r['weight_bytes']>0 and r['input_bytes']==8192*90*4 and r['output_bytes']==4096*90*4 for r in comparisons)
            assert all(comparisons[0][k]==1 for k in ('weight_equal','input_equal','output_equal'))
        else:assert not comparisons
        for i,(task_id,segment) in enumerate(segments):
            stem=f'replay-{i:02}'
            body=load(cell/(stem+'.response.json'))
            assert load(cell/(stem+'.request.json'))==replay['native_payload'](ids+prefix,1)
            replay['check_native'](body,94,1)
            row=score(body,0);assert row is not None
            records=parse_boundaries(segment)
            assert parse_comparisons(segment)==([comparisons[i]] if data['tracing'] else [])
            (cell/(stem+'.trace.log')).write_text(segment)
            if data['tracing']:
                assert records and records[0]['name']=='embd' and records[-1]['name']=='result_output'
                assert {'embd','norm-0','attn_norm-0','mamba2_y_add_d-0','mamba_out-0','nemotron_h_block_out-0','result_output'} <= {r['name'] for r in records}
                assert {'MUL_MAT','SSM_CONV','SSM_SCAN'} <= {r['op'] for r in records}
            else:assert not records
            samples.append(dict(stem=stem,task_id=task_id,tokens=body['tokens'],score=row))
            boundaries.append(records)
        item['responses']=samples
        item['unique_token_sequences']=len({tuple(s['tokens']) for s in samples})
        item['distinct_score_vectors']=len({json.dumps(s['score']['top'],sort_keys=True) for s in samples})
        if data['tracing']:
            signatures=[[(r['name'],r['op'],r['shape']) for r in rows] for rows in boundaries]
            assert all(s==signatures[0] for s in signatures),'Boundary order/shape changed; inspect raw traces before aligning'
            comparison=[]
            for i,record in enumerate(boundaries[0]):
                sums=[r[i]['printed_sums'] for r in boundaries]
                comparison.append(dict(name=record['name'],op=record['op'],header=record['header'],shape=record['shape'],printed_sums=sums,
                                       distinct_printed_sum_vectors=len({tuple(v) for v in sums})))
            item['boundary_comparison']=comparison
            item['first_changing_printed_boundary']=next((r for r in comparison if r['distinct_printed_sum_vectors']>1),None)
            item['boundary_count_per_request']=len(comparison)
            item['input_embedding_observed']=any(r['name']=='embd' for r in comparison)
            item['coverage_note']='Observed embedding, first-block projections/convolution/scan/D-add/normalization/output and logits. Printed sums remain lossy; identify unnamed nodes by retained op/source header and compare each chunk position separately.'
            save=ROOT/'trace-boundaries.json'
            save.write_text(json.dumps(boundaries,indent=2)+'\n')
    if summary['status']=='completed':assert set(result['cells'])=={'on','off'}
    original=load(out/'original-container.json');live=json.loads(command('docker','inspect',p['ORIGINAL']))[0]
    assert live['State']['Running'] and live['Id']==original['Id'] and live['Config']==original['Config']
    assert sorted(live['Mounts'],key=lambda m:m['Destination'])==sorted(original['Mounts'],key=lambda m:m['Destination'])
    assert set(command('docker','ps','--format','{{.Names}}').splitlines())=={p['ORIGINAL']}
    for label in summary['cells']:
        assert command('docker','inspect','-f','{{.State.Running}}','bolt-projection-repeat-r1-'+label).strip()=='false'
    assert binaries(p['ACCEPTED'])==load(out/'binaries.json')
    assert binaries(c['CANDIDATE'])==load(out/'candidate-binaries.json')
    hardware=runner['hardware']();assert hardware==load(out/'hardware-before.json')==load(out/'hardware-after.json')
    health=api['request'](p['URL'],'/health');assert health['status']=='ok'
    smoke=load(out/'restored-smoke.result.json');assert smoke['text'].strip()=='READY' and smoke['finish_reason']=='stop'
    result['restoration']=dict(verified_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_id=original['Id'],live_id=live['Id'],
        config_identical=True,all_mount_fields_identical=True,binaries_identical=True,hardware=hardware,health=health,ready_output=smoke['text'],all_experiments_stopped=True)
    (ROOT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    for label,item in result['cells'].items():
        print(label,json.dumps({k:v for k,v in item.items() if k not in ('responses','boundary_comparison')}))
    print('BOLT_TENSOR_TRACE_ANALYSIS_AND_RESTORATION_VERIFIED',flush=True)


if __name__=='__main__':main()
