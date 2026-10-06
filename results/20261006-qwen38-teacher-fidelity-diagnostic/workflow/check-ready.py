import json,urllib.request
from pathlib import Path
root=Path(__file__).resolve().parent
saved=json.loads((root/'provider-instance.json').read_text())
protocol=json.loads((root/'pilot-protocol.json').read_text())
key=(Path.home()/'.shadeform/api_key').read_text().strip()
request=urllib.request.Request('https://api.shadeform.ai/v1/instances/'+saved['id']+'/info',headers={'X-API-KEY':key})
with urllib.request.urlopen(request,timeout=30) as response: info=json.load(response)
assert info['name']==protocol['lease_name'] and info['cloud']=='scaleway' and info['status']=='active'
assert info['hourly_price']==330 and info['configuration']['num_gpus']==1
print('Fresh API confirms same owned validation H100 active; no second create.',flush=True)
