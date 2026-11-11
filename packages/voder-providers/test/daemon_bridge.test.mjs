import test from 'node:test';
import assert from 'node:assert/strict';
import {parseDaemonInput,makeGenerationRequest,toDaemonProposal,DAEMON_MODEL_PROTOCOL} from '../dist/daemon_contract.js';
import {runBridge} from '../bin/daemon-provider-bridge.mjs';
import {mkdtemp,readFile,writeFile,stat} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import http from 'node:http';
import {once} from 'node:events';

const good = {protocol:DAEMON_MODEL_PROTOCOL,runId:'r-12',instruction:'Fix one failing unit test.'};
const result = {providerId:'voder_default',model:'mock',text:'proposed patch',toolCalls:[],finishReason:'stop',usage:{inputTokens:5,outputTokens:7,totalTokens:12,source:'provider'}};

test('bridge rejects unrecognized request fields, malformed and overbudget input',()=>{
  for(const bad of [{...good,deliveryAllowed:true},{...good,runId:'../x'},{...good,instruction:'x'.repeat(33000)},{...good,foo:1},{...good,instruction:''}])
    assert.throws(()=>parseDaemonInput(bad));
  assert.equal(parseDaemonInput(good).runId,'r-12');
});
test('model proposal never grants authority, and keeps real usage, model and tool intent',()=>{
  const p=toDaemonProposal(good,{...result,toolCalls:[{id:'x',name:'patch',argumentsJson:'{"file":"a.rs"}'}]});
  assert.equal(p.executionAuthorized,false);
  assert.equal(p.deliveryAllowed,false);
  assert.equal(p.usage.totalTokens,12);
  assert.deepEqual(p.proposedToolCalls,[{id:'x',name:'patch',argumentsJson:'{"file":"a.rs"}'}]);
  assert.equal(makeGenerationRequest(good,'mock',128).messages.at(-1).content,good.instruction);
});
test('malformed model tool payload / huge response / fake usage fail closed',()=>{
  for(const changed of [{toolCalls:[{id:'x',name:'run',argumentsJson:'{}oops'}]}, {text:'a'.repeat(131073)}, {usage:{inputTokens:1,outputTokens:1,totalTokens:7,source:'provider'}}])
    assert.throws(()=>toDaemonProposal(good,{...result,...changed}));
});
test('actual bridge process over loopback provider produces a private immutable inert receipt',async(t)=>{
  let requests=0;
  const server=http.createServer(async(req,res)=>{
    requests++;
    assert.equal(req.url,'/v1/chat/completions');
    let body='';for await(const x of req) body+=x;
    const obj=JSON.parse(body);
    assert.equal(obj.stream,false);
    assert.match(obj.messages.at(-1).content,/failing unit test/);
    res.setHeader('Content-Type','application/json');
    res.end(JSON.stringify({model:'actual-model',choices:[{message:{content:'patch candidate'},finish_reason:'stop'}],usage:{prompt_tokens:10,completion_tokens:3}}));
  });
  server.listen(0,'127.0.0.1');await once(server,'listening');t.after(()=>new Promise(r=>server.close(r)));
  const root=await mkdtemp(join(tmpdir(),'voder7-'));
  const input=join(root,'request.json');const out=join(root,'result.json');
  await writeFile(input,JSON.stringify(good),{mode:0o600});
  const env={VODER_MODEL_PROVIDER_KIND:'openai',VODER_MODEL_API_ENDPOINT:`http://127.0.0.1:${server.address().port}/v1`,VODER_MODEL_API_KEY:'dummy-local-key',VODER_MODEL_NAME:'mock',VODER_MODEL_TIMEOUT_MS:'10000',VODER_MODEL_MAX_OUTPUT_TOKENS:'128'};
  // Loopback HTTP requires explicit opt-in in the Provider library: this is a security gate.
  await assert.rejects(runBridge(input,out,env),/HTTPS required/);
  assert.equal(requests,0);
  await runBridge(input,out,{...env,VODER_MODEL_ALLOW_LOOPBACK_HTTP:'1'});
  assert.equal(requests,1);
  const content=JSON.parse(await readFile(out,'utf8'));
  assert.equal(content.deliveryAllowed,false);
  assert.equal(content.executionAuthorized,false);
  assert.equal(content.proposalText,'patch candidate');
  assert.deepEqual(content.usage,{inputTokens:10,outputTokens:3,totalTokens:13});
  assert.equal((await stat(out)).mode & 0o777,0o600);
  // Idempotent safety: reusing a receipt destination may NEVER overwrite prior evidence.
  await assert.rejects(runBridge(input,out,{...env,VODER_MODEL_ALLOW_LOOPBACK_HTTP:'1'}));
  assert.equal(JSON.parse(await readFile(out,'utf8')).proposalText,'patch candidate');
});
test('model bridge CLI fails closed without secrets before any model invocation',async()=>{
  const root=await mkdtemp(join(tmpdir(),'voder7-'));
  const input=join(root,'request.json');const out=join(root,'result.json');
  await writeFile(input,JSON.stringify(good));
  await assert.rejects(runBridge(input,out,{}),/not configured/);
  await assert.rejects(stat(out));
});
test('separate Node CLI process uses stdout only for no protocol data and rejects tool authority',async(t)=>{
  const {spawn} = await import('node:child_process');
  const root=await mkdtemp(join(tmpdir(),'voder7-cli-'));
  const input=join(root,'request.json');const out=join(root,'result.json');
  await writeFile(input,JSON.stringify(good));
  const server=http.createServer((req,res)=>{
    res.end(JSON.stringify({model:'verified-model',choices:[{message:{content:'suggestion',tool_calls:[{id:'call_1',type:'function',function:{name:'patch',arguments:'{"path":"src/a.rs"}'}}]},finish_reason:'tool_calls'}],usage:{prompt_tokens:2,completion_tokens:6}}));
  });
  server.listen(0,'127.0.0.1');await once(server,'listening');t.after(()=>new Promise(r=>server.close(r)));
  const env={...process.env,VODER_MODEL_PROVIDER_KIND:'openai',VODER_MODEL_API_ENDPOINT:`http://127.0.0.1:${server.address().port}/v1`,VODER_MODEL_ALLOW_LOOPBACK_HTTP:'1',VODER_MODEL_API_KEY:'NEVER_PRINT_THIS_TEST_SECRET',VODER_MODEL_NAME:'mock',VODER_MODEL_TIMEOUT_MS:'10000',VODER_MODEL_MAX_OUTPUT_TOKENS:'128'};
  const child=spawn(process.execPath,[new URL('../bin/daemon-provider-bridge.mjs',import.meta.url).pathname,input,out],{env,stdio:['ignore','pipe','pipe']});
  const chunks=[]; const errors=[];
  child.stdout.on('data',x=>chunks.push(x));child.stderr.on('data',x=>errors.push(x));
  const [code]=await once(child,'exit');
  assert.equal(code,0,'Node CLI actual subprocess must succeed');
  assert.equal(Buffer.concat(chunks).length,0,'stdout must never carry secret or result');
  assert.doesNotMatch(Buffer.concat(errors).toString(),/NEVER_PRINT_THIS_TEST_SECRET/);
  const value=JSON.parse(await readFile(out,'utf8'));
  assert.equal(value.model,'verified-model');
  assert.equal(value.executionAuthorized,false);
  assert.equal(value.deliveryAllowed,false);
  assert.equal(value.proposedToolCalls[0].name,'patch');
});
