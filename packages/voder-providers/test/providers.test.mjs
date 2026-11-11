import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import {once} from 'node:events';
import {OpenAICompatibleProvider, AnthropicProvider, ProviderRegistry, ProviderError, UsageLedger, resolveConfiguration, normalizeUsage, validateRequest} from '../dist/index.js';

const msg = [{role:'user',content:'fix failing test'}];
const request = (overrides = {}) => ({model:'mock-model',messages:msg,maxOutputTokens:64,...overrides});
const chunk = (body) => `data: ${JSON.stringify(body)}\n\n`;
const sse = (...items) => `${items.map(chunk).join('')}data: [DONE]\n\n`;
const finish = (text) => ({model:'mock-model', choices:[{delta:{content:text},finish_reason:null}]});
const completed = {model:'mock-model',choices:[{delta:{},finish_reason:'stop'}]};
async function server(handler) {
  const requests = [];
  const instance = http.createServer(async (req,res) => {
    let body=''; for await (const part of req) body += part.toString();
    let json;try{json=JSON.parse(body)}catch{}
    requests.push({req,json});
    try {await handler(req,res,json);} catch(e){res.writeHead(500);res.end('server test failure');}
  });
  instance.listen(0,'127.0.0.1'); await once(instance,'listening');
  const port = instance.address().port;
  return {base: `http://127.0.0.1:${port}`, requests, close: () => new Promise(resolve => instance.close(resolve))};
}
const openai = (s, opts={}) => new OpenAICompatibleProvider({id:'local',kind:'openai',endpoint:s.base+'/v1',allowLoopbackHttp:true,...opts});
const anthropic = (s, opts={}) => new AnthropicProvider({id:'claude',kind:'anthropic',endpoint:s.base,allowLoopbackHttp:true,apiKey:'test-key',...opts});
async function collect(p, r=request()) {let values=[];for await (let v of p.stream(r)) values.push(v);return values;}
function expectCode(code) {return e => e instanceof ProviderError && e.code === code;}

test('OpenAI: model text, usage and safe /v1 path',async(t)=>{
 const s=await server((req,res,body)=>{assert.equal(req.url,'/v1/chat/completions');assert.equal(body.model,'mock-model');assert.equal(body.stream,false);res.setHeader('Content-Type','application/json');res.end(JSON.stringify({model:'actual-model',choices:[{message:{content:'fixed'},finish_reason:'stop'}],usage:{prompt_tokens:5,completion_tokens:7}}));});t.after(s.close);
 const result=await openai(s).generate(request());assert.equal(result.model,'actual-model');assert.equal(result.text,'fixed');assert.deepEqual(result.usage,{inputTokens:5,outputTokens:7,totalTokens:12,source:'provider'});
});
test('OpenAI stream: partial fragmented frames, usage and actual finish',async(t)=>{
 const s=await server(async(req,res,body)=>{assert.equal(body.stream,true);res.writeHead(200,{'content-type':'text/event-stream'}); const frames=sse(finish('he'),finish('llo'),completed,{model:'mock-model',choices:[],usage:{prompt_tokens:3,completion_tokens:2}});for(let i=0;i<frames.length;i+=7)res.write(frames.slice(i,i+7));res.end();});t.after(s.close);
 const events=await collect(openai(s));assert.deepEqual(events.filter(e=>e.kind==='text_delta').map(e=>e.text),['he','llo']);assert.equal(events.at(-1).result.text,'hello');assert.equal(events.at(-1).result.usage.totalTokens,5);
});
test('OpenAI stream tools are inert intent and assembled correctly',async(t)=>{
 const s=await server((req,res)=>{res.writeHead(200,{'content-type':'text/event-stream'});res.end(sse({model:'mock-model',choices:[{delta:{tool_calls:[{index:0,id:'call-1',function:{name:'inspect',arguments:'{"path":'}}]},finish_reason:null}]},{model:'mock-model',choices:[{delta:{tool_calls:[{index:0,function:{arguments:'"a.rs"}'}}]},finish_reason:null}]},{model:'mock-model',choices:[{delta:{},finish_reason:'tool_calls'}]}));});t.after(s.close);
 const events=await collect(openai(s));assert.deepEqual(events.at(-1).result.toolCalls,[{id:'call-1',name:'inspect',argumentsJson:'{"path":"a.rs"}'}]);assert.equal(events.at(-1).result.finishReason,'tool_calls');
});
test('OpenAI malformed tool arguments fail closed',async(t)=>{
 const s=await server((req,res)=>{res.end(JSON.stringify({model:'mock-model',choices:[{message:{content:null,tool_calls:[{id:'1',function:{name:'exec',arguments:'{broken'}}]},finish_reason:'tool_calls'}]}));});t.after(s.close);
 await assert.rejects(openai(s).generate(request()),expectCode('MALFORMED_RESPONSE'));
});
test('OpenAI truncation without DONE is rejected',async(t)=>{
 const s=await server((req,res)=>{res.writeHead(200,{'content-type':'text/event-stream'});res.end(chunk(finish('partial')));});t.after(s.close);
 await assert.rejects(collect(openai(s)),expectCode('MALFORMED_RESPONSE'));
});
test('OpenAI malformed JSON stream is rejected',async(t)=>{
 const s=await server((req,res)=>{res.writeHead(200,{'content-type':'text/event-stream'});res.end('data: {invalid}\n\n');});t.after(s.close);
 await assert.rejects(collect(openai(s)),expectCode('MALFORMED_RESPONSE'));
});
test('HTTP status sanitization and Retry-After',async(t)=>{
 const s=await server((req,res)=>{res.writeHead(429,{'retry-after':'5'});res.end('secret-api-key');});t.after(s.close);
 await assert.rejects(openai(s).generate(request()),e=>e.code==='RATE_LIMITED'&&e.retryable&&e.retryAfterMs===5000&&!e.message.includes('secret'));
});
test('HTTPS enforced except explicit loopback',()=>{
 assert.throws(()=>new OpenAICompatibleProvider({id:'x',kind:'openai',endpoint:'http://example.org/v1'}),expectCode('INVALID_CONFIGURATION'));
 assert.throws(()=>new OpenAICompatibleProvider({id:'x',kind:'openai',endpoint:'http://127.0.0.1:8000/v1'}),expectCode('INVALID_CONFIGURATION'));
 assert.throws(()=>new OpenAICompatibleProvider({id:'x',kind:'openai',endpoint:'https://user:pw@example.org'}),expectCode('INVALID_CONFIGURATION'));
 assert.throws(()=>new OpenAICompatibleProvider({id:'x',kind:'openai',endpoint:'https://x.org/?k=1'}),expectCode('INVALID_CONFIGURATION'));
 assert.equal(resolveConfiguration({id:'x',kind:'openai',endpoint:'https://test.org'}).endpoint.host,'test.org');
});
test('model request budgets reject negative, massive or unsafe data',()=>{
 for(const r of [request({maxOutputTokens:0}),request({maxOutputTokens:131073}),request({messages:[]}),request({messages:[{role:'tool',content:'data'}]}),request({temperature:-1}),request({messages:[{role:'user',content:'x'.repeat(2*1024*1024+1)}]})]) assert.throws(()=>validateRequest(r));
});
test('response size limit stops unbounded provider output',async(t)=>{
 const s=await server((req,res)=>res.end('a'.repeat(20_000)));t.after(s.close);
 await assert.rejects(openai(s,{maxResponseBytes:1024}).generate(request()),expectCode('RESPONSE_TOO_LARGE'));
});
test('no automatic retry on ambiguous network/outcome error',async(t)=>{
 const s=await server((req,res)=>{res.writeHead(503);res.end('down');});t.after(s.close);
 await assert.rejects(openai(s).generate(request()),expectCode('UPSTREAM_UNAVAILABLE'));assert.equal(s.requests.length,1);
});
test('ProviderRegistry forbids duplicate ids',async(t)=>{
 const s=await server((req,res)=>res.end('{}'));t.after(s.close);
 const registry = new ProviderRegistry();registry.register(openai(s));assert.deepEqual(registry.list(),['local']);assert.throws(()=>registry.register(openai(s)));
});
test('UsageLedger never invents token usage or model prices',()=>{
 const ledger=new UsageLedger();ledger.record({inputTokens:200,outputTokens:100,totalTokens:300,source:'provider'},{inputUsd:1,outputUsd:2});assert.equal(ledger.snapshot().costUsd,0.0004);ledger.record(undefined);assert.equal(ledger.snapshot().costUsd,undefined);assert.equal(ledger.snapshot().reported,1);assert.equal(normalizeUsage({input_tokens:3,output_tokens:4}).totalTokens,7);
});
test('Anthropic ordinary generate uses correct wire contract',async(t)=>{
 const s=await server((req,res,body)=>{assert.equal(req.url,'/v1/messages');assert.equal(req.headers['x-api-key'],'test-key');assert.equal(body.system,'Follow rules');assert.deepEqual(body.messages,[{role:'user',content:'hi'}]);res.end(JSON.stringify({model:'actual-claude',content:[{type:'text',text:'answer'}],stop_reason:'end_turn',usage:{input_tokens:5,output_tokens:9}}));});t.after(s.close);
 const result=await anthropic(s).generate(request({messages:[{role:'system',content:'Follow rules'},{role:'user',content:'hi'}]}));assert.equal(result.text,'answer');assert.equal(result.usage.totalTokens,14);
});
test('Anthropic streaming text and final usage',async(t)=>{
 const s=await server((req,res)=>{res.writeHead(200,{'content-type':'text/event-stream'});res.end([['message_start',{type:'message_start',message:{model:'claude-tested',usage:{input_tokens:6,output_tokens:0}}}],['content_block_start',{type:'content_block_start',index:0,content_block:{type:'text',text:''}}],['content_block_delta',{type:'content_block_delta',index:0,delta:{type:'text_delta',text:'okay'}}],['message_delta',{type:'message_delta',delta:{stop_reason:'end_turn'},usage:{output_tokens:2}}],['message_stop',{type:'message_stop'}]].map(([name,val])=>`event: ${name}\ndata: ${JSON.stringify(val)}\n\n`).join(''));});t.after(s.close);
 const events=await collect(anthropic(s));assert.equal(events.at(-1).result.text,'okay');assert.equal(events.at(-1).result.usage.totalTokens,8);
});
test('Anthropic tool results are not silently reinterpreted',async(t)=>{
 const s=await server((req,res)=>res.end('{}'));t.after(s.close);
 await assert.rejects(anthropic(s).generate(request({messages:[{role:'tool',content:'result',toolCallId:'id'}]})),expectCode('UNSUPPORTED'));
 assert.equal(s.requests.length,0);
});
test('Anthropic missing final marker rejected',async(t)=>{
 const s=await server((req,res)=>{res.writeHead(200,{'content-type':'text/event-stream'});res.end(chunk({type:'message_start',message:{model:'x',usage:{input_tokens:2,output_tokens:0}}}));});t.after(s.close);
 await assert.rejects(collect(anthropic(s)),expectCode('MALFORMED_RESPONSE'));
});
test('timeout fails without retry',async(t)=>{
 const s=await server(()=>{});t.after(s.close);
 let attempts=0;
 const fetchImpl=(url,options)=>{attempts++;return fetch(url,options);};
 await assert.rejects(openai(s,{timeoutMs:100,fetchImpl}).generate(request()),expectCode('TIMEOUT'));
 assert.equal(attempts,1);
});
test('pre-cancelled request fails and does not reach service',async(t)=>{
 const s=await server((req,res)=>res.end('{}'));t.after(s.close);
 const controller=new AbortController();controller.abort();await assert.rejects(openai(s).generate(request({signal:controller.signal})),expectCode('CANCELLED'));assert.equal(s.requests.length,0);
});
test('provider exceptions never reproduce Authorization secrets',async()=>{
 const p=new OpenAICompatibleProvider({id:'xx',kind:'openai',endpoint:'https://test.invalid/v1',apiKey:'CANARY-PRIVATE',fetchImpl:async()=>{throw Error('CANARY-PRIVATE');}});
 await assert.rejects(p.generate(request()),e=>e.code==='NETWORK'&&!e.message.includes('CANARY-PRIVATE'));
});
test('timeout during streaming body maps to TIMEOUT, not malformed response',async(t)=>{
 const s=await server(async(req,res)=>{res.writeHead(200,{'content-type':'text/event-stream'});res.write(chunk(finish('start')));await new Promise(resolve=>setTimeout(resolve,180));res.end(sse(completed));});t.after(s.close);
 await assert.rejects(collect(openai(s,{timeoutMs:35})),expectCode('TIMEOUT'));
});
test('cancel after beginning the stream reports CANCELLED',async(t)=>{
 const s=await server(async(req,res)=>{res.writeHead(200,{'content-type':'text/event-stream'});res.write(chunk(finish('start')));await new Promise(resolve=>setTimeout(resolve,180));res.end(sse(completed));});t.after(s.close);
 const controller=new AbortController();
 const stream=collect(openai(s),request({signal:controller.signal}));
 setTimeout(()=>controller.abort(),35);
 await assert.rejects(stream,expectCode('CANCELLED'));
});
test('UsageLedger invalid usage/price must not mutate counters',()=>{
 const ledger = new UsageLedger();const value={inputTokens:1,outputTokens:2,totalTokens:3,source:'provider'};
 assert.throws(()=>ledger.record(value,{inputUsd:-1,outputUsd:3}));
 assert.throws(()=>ledger.record({...value,inputTokens:-1}));
 assert.deepEqual(ledger.snapshot(),{requests:0,reported:0,inputTokens:0,outputTokens:0,costUsd:0});
});
test('OpenAI redirects fail and do not forward authorization',async(t)=>{
 const s=await server((req,res)=>{res.writeHead(302,{location:'http://example.net/steal'});res.end();});t.after(s.close);
 await assert.rejects(openai(s,{apiKey:'SECRET'}).generate(request()), e=>e instanceof ProviderError&&e.code==='UNSAFE_REDIRECT');
 assert.equal(s.requests.length,1);
});
test('Anthropic SSE tool call and fragmented JSON arguments',async(t)=>{
 const frames=[{type:'message_start',message:{model:'m',usage:{input_tokens:5,output_tokens:0}}},{type:'content_block_start',index:0,content_block:{type:'tool_use',id:'tool-01',name:'read_file',input:{}}},{type:'content_block_delta',index:0,delta:{type:'input_json_delta',partial_json:'{"file":'}},{type:'content_block_delta',index:0,delta:{type:'input_json_delta',partial_json:'"abc.py"}'}},{type:'content_block_stop',index:0},{type:'message_delta',delta:{stop_reason:'tool_use'},usage:{output_tokens:6}},{type:'message_stop'}];
 const s=await server((req,res)=>{res.writeHead(200,{'content-type':'text/event-stream'});res.end(frames.map(x=>`event: ${x.type}\ndata: ${JSON.stringify(x)}\n\n`).join(''));});t.after(s.close);
 const events=await collect(anthropic(s));assert.equal(events.at(-1).result.finishReason,'tool_calls');assert.deepEqual(events.at(-1).result.toolCalls,[{id:'tool-01',name:'read_file',argumentsJson:'{"file":"abc.py"}'}]);
});
test('model provider cannot mark pass or access arbitrary tools via accepted request',async(t)=>{
 const s=await server((req,res,body)=>{assert.equal(body?.messages?.[0]?.role,'user');assert.equal(body?.command,undefined);res.end(JSON.stringify({model:'m',choices:[{message:{content:'test code passed'},finish_reason:'stop'}]}));});t.after(s.close);
 const provider=openai(s);assert.equal(typeof provider['executeTool'],'undefined');assert.equal(typeof provider['approve'],'undefined');const r=await provider.generate(request());assert.equal(r.text,'test code passed');assert.equal('authorized' in r,false);
});
test('OpenAI compatible vLLM local instance can be configured without API key',async(t)=>{
 const s=await server((req,res)=>{assert.equal(req.headers.authorization,undefined);res.end(JSON.stringify({model:'Qwen',choices:[{message:{content:'yes'},finish_reason:'stop'}]}));});t.after(s.close);
 const result=await openai(s).generate(request());assert.equal(result.model,'Qwen');
});
test('SSE randomized chunk boundaries keep Unicode intact (500 deterministic cases)',async()=>{
 const {parseSse}=await import('../dist/sse.js');
 const raw=': keepalive\r\ndata: {"msg":"中✨文"}\r\n\r\ndata: [DONE]\r\n\r\n';
 const bytes=new TextEncoder().encode(raw);
 let state=918273;
 for(let trial=0;trial<500;trial++) {
  const step=()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return 1+(state%19);};
  let offset=0;const stream=new ReadableStream({pull(controller){if(offset>=bytes.length){controller.close();return;}const n=step();controller.enqueue(bytes.slice(offset,offset+n));offset+=n;}});
  const frames=[];for await (const frame of parseSse(new Response(stream),4096))frames.push(frame.data);
  assert.deepEqual(frames,['{"msg":"中✨文"}','[DONE]']);
 }
});
test('SSE deterministic malformed UTF-8 rejected',async()=>{
 const {parseSse}=await import('../dist/sse.js');
 const stream=new ReadableStream({start(c){c.enqueue(new Uint8Array([100,97,116,97,58,32,0xff,10,10]));c.close();}});
 await assert.rejects(async()=>{for await(const f of parseSse(new Response(stream),4096)){}},expectCode('MALFORMED_RESPONSE'));
});
test('provider upstream error event fails safely without being a tool command',async(t)=>{
 const s=await server((req,res)=>{res.writeHead(200,{'content-type':'text/event-stream'});res.end(chunk({error:{message:'do not expose underlying API key'}}));});t.after(s.close);
 await assert.rejects(collect(openai(s)),e=>e.code==='UPSTREAM_UNAVAILABLE'&&!e.message.includes('API key'));
});
test('Anthropic nonstream tool_use returned as suggestion only',async(t)=>{
 const s=await server((req,res)=>res.end(JSON.stringify({model:'claude',content:[{type:'tool_use',id:'id1',name:'edit',input:{file:'a.txt'}}],stop_reason:'tool_use'})));t.after(s.close);
 const r=await anthropic(s).generate(request());assert.deepEqual(r.toolCalls,[{id:'id1',name:'edit',argumentsJson:'{"file":"a.txt"}'}]);assert.equal('permission' in r,false);
});
test('out-of-range tool call delta index rejected',async(t)=>{
 const s=await server((req,res)=>{res.writeHead(200,{'content-type':'text/event-stream'});res.end(sse({model:'model',choices:[{delta:{tool_calls:[{index:200,id:'bad',function:{name:'tool',arguments:'{}'}}]},finish_reason:'tool_calls'}]}));});t.after(s.close);
 await assert.rejects(collect(openai(s)),expectCode('MALFORMED_RESPONSE'));
});
test('duplicate Anthropic stream content_block index rejected',async(t)=>{
 const f=[{type:'message_start',message:{model:'m',usage:{input_tokens:1,output_tokens:0}}},{type:'content_block_start',index:0,content_block:{type:'text',text:''}},{type:'content_block_start',index:0,content_block:{type:'text',text:''}}];
 const s=await server((req,res)=>{res.writeHead(200,{'content-type':'text/event-stream'});res.end(f.map(x=>chunk(x)).join(''));});t.after(s.close);
 await assert.rejects(collect(anthropic(s)),expectCode('MALFORMED_RESPONSE'));
});
