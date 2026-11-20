import json,sys
def load(p): return {json.loads(l)['case']:json.loads(l) for l in open(p)}
A=load(sys.argv[1]); B=load(sys.argv[2])
ident=0; bit=0
for c in sorted(A):
    a,b=A[c]['token_ids'],B[c]['token_ids']
    fd=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),None)
    n=fd if fd is not None else len(a)
    dl=max((abs(x-y) for x,y in zip(A[c]['lp'][:n],B[c]['lp'][:n])),default=0)
    ident+= fd is None; bit += (fd is None and A[c]['lp']==B[c]['lp'] and A[c]['top']==B[c]['top'])
    print(c,'first_diff',fd,'maxdlp_before',round(dl,5))
print('token-identical',ident,'/',len(A),' bitwise (tokens+lp+top)',bit)
