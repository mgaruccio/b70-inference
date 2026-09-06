import json,math,re,statistics,subprocess,sys,threading,time,urllib.request
from pathlib import Path
C=int(sys.argv[1]);D=Path(sys.argv[2]);CLIENT=Path('/home/mike/b70-evals/muse-glimmer/20260905-concurrency/instrument.py')
BASE='http://127.0.0.1:18080'
def get_metrics():
 with urllib.request.urlopen(BASE+'/metrics',timeout=10) as response:s=response.read().decode()
 result={'t':time.monotonic()}
 for key in ['num_requests_running','num_requests_waiting','kv_cache_usage_perc','num_preemptions_total']:
  values=re.findall(r'^vllm:'+key+r'\{[^\n]*?\}\s+(\S+)',s,re.M)
  if not values:raise RuntimeError('Missing metric '+key)
  result[key]=sum(map(float,values))
 return result,s
with urllib.request.urlopen(BASE+'/v1/models',timeout=10) as response:models=json.load(response)
assert models['data'][0]['max_model_len']==131072
(D/'models.json').write_text(json.dumps(models,indent=2))
common=[sys.executable,str(CLIENT),'--base',BASE+'/v1','--model','muse-glimmer-gptq','--concurrency',str(C),'--max-tokens','256','--log','']
subprocess.run(common+['--reps','1','--label',f'c{C}-warm','--out',str(D/'warm.json')],check=True)
before,raw=get_metrics();(D/'metrics-before.txt').write_text(raw);samples=[before];stop=threading.Event()
def monitor():
 while not stop.wait(.2):
  try:samples.append(get_metrics()[0])
  except Exception as exc:samples.append({'t':time.monotonic(),'error':repr(exc)})
thread=threading.Thread(target=monitor,daemon=True);thread.start()
try:run=subprocess.run(common+['--reps','5','--label',f'c{C}','--out',str(D/'throughput.json')])
finally:stop.set();thread.join()
after,raw=get_metrics();samples.append(after);(D/'metrics-after.txt').write_text(raw)
(D/'scheduler-metrics.json').write_text(json.dumps(samples,indent=2));assert run.returncode==0
j=json.loads((D/'throughput.json').read_text());waves=j['waves'];rows=[row for w in waves for row in w['rows']]
assert len(waves)==5 and len(rows)==5*C and all(not w['errors'] and w['ok']==C for w in waves)
assert all(row['ok'] and row['completion_tokens']==256 and row.get('reasoning_chars',0)+row.get('content_chars',0)>0 for row in rows)
valid=[s for s in samples if 'num_requests_running' in s]
def median(key):return statistics.median(row[key] for row in rows if row.get(key) is not None)
def p95(key):
 values=sorted(row[key] for row in rows if row.get(key) is not None);return values[math.ceil(.95*len(values))-1]
rates=[w['aggregate_e2e_tok_s'] for w in waves]
summary={'concurrency':C,'wave_rates':rates,'median_aggregate_tok_s':statistics.median(rates),'median_ttft_s':median('ttft_s'),'p95_ttft_s':p95('ttft_s'),'median_request_s':median('elapsed_s'),'p95_request_s':p95('elapsed_s'),'median_client_decode_tok_s':median('decode_tok_s'),'peak_scheduler_running':max(s['num_requests_running'] for s in valid),'peak_waiting':max(s['num_requests_waiting'] for s in valid),'peak_kv_fraction':max(s['kv_cache_usage_perc'] for s in valid),'preemptions':after['num_preemptions_total']-before['num_preemptions_total'],'drained':after['num_requests_running']==after['num_requests_waiting']==0,'successful_streams':len(rows),'metric_errors':sum('error' in s for s in samples)}
(D/'summary.json').write_text(json.dumps(summary,indent=2));print('SWEEP_SUMMARY '+json.dumps(summary),flush=True)
assert summary['peak_scheduler_running']==C and summary['preemptions']==0 and summary['drained']
