"""Small synthetic checks for structural classification; no generated tools run."""
import copy
import check_outputs as checks
request={'tools':[{'function':{'name':'bash','parameters':{'type':'object','properties':{'command':{'type':'string'},'timeout':{'type':'number'}},'required':['command'],'additionalProperties':False}}}]}
base={'choices':[{'finish_reason':'tool_calls','message':{'role':'assistant','tool_calls':[{'type':'function','function':{'name':'bash','arguments':'{"command":"echo synthetic"}'}}]}}]}
assert checks.inspect(request,base)['invalid_calls']==0
for arguments,key in [('{','argument_json_errors'),('{"command":1}','argument_schema_errors'),('{"command":"x","extra":true}','argument_schema_errors'),('{"command":"x","timeout":true}','argument_schema_errors'),('{"command":"x","timeout":NaN}','argument_json_errors')]:
 response=copy.deepcopy(base);response['choices'][0]['message']['tool_calls'][0]['function']['arguments']=arguments
 result=checks.inspect(request,response);assert result[key]==1 and result['invalid_calls']==1 and result['invalid_call_requests']==1
response=copy.deepcopy(base);response['choices'][0]['message']['tool_calls'][0]['function']['name']='unknown'
assert checks.inspect(request,response)['unknown_tool_or_type']==1
response={'choices':[{'finish_reason':'length','message':{'role':'assistant','tool_calls':None,'content':'synthetic'}}]}
result=checks.inspect(request,response);assert result['finish_length']==1 and result['tool_calls']==0
print('8 synthetic output-structure checks passed; no tool execution.')
