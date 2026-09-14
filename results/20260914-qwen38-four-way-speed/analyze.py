#!/usr/bin/env python3
"""Analyze only this matched eight-cell campaign, without historical pooling."""
import json
import statistics as st
from pathlib import Path
ROOT = Path(__file__).resolve().parent
ARMS = ('mtp4', 'custom', 'dflash', 'dspark')
LENGTHS = (512, 8192, 32768, 65536)

def load(path):
    return json.loads(path.read_text())

def stats(values):
    q = st.quantiles(values, n=4, method='inclusive')
    return {'median': st.median(values), 'iqr': q[2]-q[0], 'samples': values}

result = {'tier': 'development', 'historical_samples_included': False,
          'order': ['mtp4-01','custom-01','dflash-01','dspark-01','dspark-02','dflash-02','custom-02','mtp4-02'],
          'excluded_samples': 0, 'points': {}, 'cells': {}}
for arm in ARMS:
    for rep in (1,2):
        name = f'{arm}-{rep:02}'
        s = load(ROOT/name/'summary.json')
        assert s['status']=='passed' and s['host_unchanged'], name
        assert s['benchmark']['gates']['status']=='passed', name
        if arm == 'custom':
            e=s['benchmark']['candidate_execution']
            assert e['eligible_dispatch_log_count']>0 and e['unsupported_q5_log_count']==0
            assert e['full_graph_capture_seen'] and e['full_graph_run_seen']
        result['cells'][name] = {'status':s['status'],'host_unchanged':s['host_unchanged'],
            'launch_metadata':load(ROOT/name/'launch-metadata.json')}
for length in LENGTHS:
    arms={}
    for arm in ARMS:
        all_rows=[]; cell_medians=[]
        for rep in (1,2):
            p=ROOT/f'{arm}-{rep:02}'/f'length-{length}'/'long-context'
            point=load(p/'summary.json')['points'][0]
            rows=point['measurements']
            assert len(rows)==6
            for i,row in enumerate(rows,1):
                v=row['validation']
                assert row['valid'] and v['prompt_tokens']==length and v['completion_tokens']==128
                suffix=f'length-{length}/long-context/points/length-{length}/measured-{i:02}/request.json'
                assert load(ROOT/f'{arm}-{rep:02}'/suffix)==load(ROOT/'mtp4-01'/suffix)
            all_rows.extend(rows)
            cell_medians.append(st.median(r['decode_tps_post_first'] for r in rows))
        arms[arm]={'decode_tps':stats([r['decode_tps_post_first'] for r in all_rows]),
                   'ttft_s':stats([r['ttft_s'] for r in all_rows]),
                   'total_time_s':stats([r['total_time_s'] for r in all_rows]),
                   'cell_decode_medians':cell_medians}
    for arm in ARMS:
        arms[arm]['delta_vs_native_percent']=100*(arms[arm]['decode_tps']['median']/arms['mtp4']['decode_tps']['median']-1)
    result['points'][str(length)]=arms
print(json.dumps(result,indent=2))
