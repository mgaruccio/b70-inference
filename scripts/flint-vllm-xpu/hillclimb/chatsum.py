import json,statistics as st,sys
for f in sys.argv[1:]:
    rows=[json.loads(l) for l in open(f)]
    out=[]
    for mode in ('greedy','service'):
        r=[x for x in rows if x['sampling']==mode and not x['warmup']]
        tps=st.median(x['decode_tps'] for x in r)
        al=''
        if r[0]['spec']:
            d=sum(x['spec']['drafts'] for x in r); a=sum(x['spec']['accepted'] for x in r); al='AL %.2f'%(1+a/d)
        out.append(f'{mode} {tps:.1f} {al}')
    print(f.split('/')[-1], ' | '.join(out))
