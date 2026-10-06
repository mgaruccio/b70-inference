import configparser,json,os,urllib.request
from pathlib import Path
import boto3
from botocore.config import Config
os.umask(0o077)
root=Path(__file__).resolve().parent
config=configparser.RawConfigParser();config.read(Path.home()/'.config/rclone/rclone.conf');r=config['r2']
s3=boto3.client('s3',endpoint_url=r['endpoint'],aws_access_key_id=r['access_key_id'],aws_secret_access_key=r['secret_access_key'],region_name='auto',config=Config(signature_version='s3v4'))
prefix='2026-10-06/cache/b70-evals/20261006-qwen38-serving-recurrence-validation'
phases=['bootstrap','validation','evaluation','final']
names=['preflight.json','tuned-mtp.safetensors','selected-mtp.safetensors']
for phase in phases:
    names += [phase+'.tgz',phase+'.tgz.parts.json']+[phase+'.tgz.part'+str(i).zfill(2) for i in range(32)]
items=[{'path':name,'url':s3.generate_presigned_url('put_object',Params={'Bucket':'ml-archive','Key':prefix+'/'+name},ExpiresIn=21600)} for name in names]
with (root/'backup-private.json').open('x') as f:json.dump(items,f)
probe=json.dumps({'phase':'local_upload_preflight','run':'qwen38-full-corpus-cuda','no_reduced_corpus':True}).encode()
with urllib.request.urlopen(urllib.request.Request(items[0]['url'],data=probe,method='PUT',headers={'Content-Length':str(len(probe))}),timeout=30) as response:assert response.status in (200,201,204)
assert s3.get_object(Bucket='ml-archive',Key=prefix+'/preflight.json')['Body'].read()==probe
(root/'r2-location.txt').write_text('r2:ml-archive/'+prefix+'\n')
source_prefix='2026-10-05/cache/b70-evals/20261005-qwen38-full-corpus-kl-cuda-retry1'
index=json.loads((root/'source-final.tgz.parts.json').read_text())
assert index['sha256']=='27ba5a05d762c90a6fbba164486dce636dac125214fb62ca40aaaad17c950458'
restore={**index,'parts':[{**part,'url':s3.generate_presigned_url('get_object',Params={'Bucket':'ml-archive','Key':source_prefix+'/'+part['path']},ExpiresIn=21600)} for part in index['parts']]}
with (root/'restore-private.json').open('x') as f:json.dump(restore,f)
print('R2 PUT/readback passed; bounded archive-part URLs staged,no cloud credentials exported.',flush=True)
