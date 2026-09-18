"""Check private API outputs structurally; never execute tools or infer task success."""
import collections,json,pathlib,sys
import jsonschema

def inspect(request,response):
    counts=collections.Counter(requests=1)
    choice=response['choices'][0];message=choice['message']
    counts['finish_'+choice['finish_reason']]+=1
    if message.get('role')!='assistant':counts['role_errors']+=1
    tools={t['function']['name']:t['function']['parameters'] for t in request.get('tools',[])}
    calls=message.get('tool_calls') or []
    if calls:counts['tool_call_requests']+=1
    invalid_request=False
    for call in calls:
        counts['tool_calls']+=1;invalid=False
        function=call.get('function',{})
        if call.get('type')!='function' or function.get('name') not in tools:
            counts['unknown_tool_or_type']+=1;invalid=True
        else:
            try:
                def reject_constant(value):raise ValueError('Non-JSON numeric constant')
                arguments=json.loads(function['arguments'],parse_constant=reject_constant)
            except (KeyError,TypeError,ValueError):
                counts['argument_json_errors']+=1;invalid=True
            else:
                try:jsonschema.validate(arguments,tools[function['name']])
                except jsonschema.ValidationError:counts['argument_schema_errors']+=1;invalid=True
        if invalid:counts['invalid_calls']+=1;invalid_request=True
    if invalid_request:counts['invalid_call_requests']+=1
    return counts

def main(root):
    expected=json.loads((root/'admission-summary.json').read_text())['selected_contexts']
    admission=json.loads((root/'admission-private.json').read_text())
    info={r['id']:r for r in admission['records']}
    runs={};reference=[]
    for label in ['A1','B1','B2','A2']:
        counts=collections.Counter()
        paths=sorted((root/f'test-{label}-output').glob('request-*/response.json'))
        assert len(paths)==expected
        for index,path in enumerate(paths):
            response=json.loads(path.read_text());request=json.loads((path.parent/'request.json').read_text())
            measurement=json.loads((path.parent/'measurement.json').read_text())
            identifier=admission['ordered_selected_ids'][index]
            assert measurement['prompt_id']==identifier
            assert len(response['prompt_token_ids'])==info[identifier]['prompt_tokens']
            assert request['max_tokens']==info[identifier]['max_tokens']
            assert request['temperature']==0 and request['seed']==42
            counts.update(inspect(request,response))
            normalized={k:v for k,v in request.items() if k not in ['request_id','cache_salt']}
            signature=(normalized,response['prompt_token_ids'])
            if label=='A1':reference.append(signature)
            else:assert signature==reference[index]
        runs[label]=dict(counts)
    pooled={}
    for side,labels in [('stock',['A1','A2']),('candidate',['B1','B2'])]:
        counts=collections.Counter()
        for label in labels:counts.update(runs[label])
        pooled[side]=dict(counts)
    result={'scope':'output structure only; no functional quality or semantic-equivalence claim', 'requests_and_prompt_tokens_match':4*expected,'runs':runs,'pooled':pooled}
    with (root/'output-checks.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main(pathlib.Path(sys.argv[1]))
