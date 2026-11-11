import {test} from 'node:test';
import assert from 'node:assert/strict';
import {guardedStream} from '../src/request-policy.mjs';
import {mkdtemp,writeFile,mkdir,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {execFileSync} from 'node:child_process';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
const model={provider:'voder-model-api',id:'deepseek-flash',baseUrl:'https://api.deepseek.com/v1',contextWindow:65536,maxTokens:4096};
function guard(m=model){let calls=0,audit;return {run:guardedStream({raw:()=>{calls++;return {};},normalizeContext:x=>x,systemPrompt:'RULES',inspect(){},allowedModels:[m],currentModel:()=>m,onAudit:x=>{audit=x;}}),get calls(){return calls;},get audit(){return audit;}};}
const context=n=>({messages:[{role:'system',content:'RULES'},{role:'user',content:Array.from({length:n},()=>({type:'image',mimeType:'image/png',data:'AAAA'}))}]});
test('official Flash uses its documented image token bound, not 16K per image',()=>{const g=guard();g.run(model,context(50),{});assert.equal(g.calls,1);assert.equal(g.audit.image_token_allowance,51200);});
test('unknown/proxy model never inherits Flash token assumptions',()=>{const m={...model,baseUrl:'https://api.deepseek.com.example.invalid',contextWindow:1048576},g=guard(m);g.run(m,context(2),{});assert.equal(g.audit.image_token_allowance,32768);});
test('provider image count rejects clearly before a network call',()=>{const g=guard();assert.throws(()=>g.run(model,context(601),{}),{code:'MODEL_IMAGE_CAPACITY'});assert.equal(g.calls,0);});
test('provider body capacity rejects without dropping image bytes',()=>{const g=guard(),c=context(1);c.messages[1].content[0].data='A'.repeat(49*1024*1024);assert.throws(()=>g.run(model,c,{}),{code:'MODEL_INPUT_CAPACITY'});assert.equal(g.calls,0);});
test('official Flash default modality applies only to its documented official identity',async()=>{
 const home=await mkdtemp(join(tmpdir(),'voder-modality-test-'));
 try{
  await mkdir(join(home,'.config/voder'),{recursive:true});const config=join(home,'.config/voder/model-provider.json');
  const module=resolve(new URL('../../voder-vscode/src/model-env.cjs',import.meta.url).pathname);
  const probe=()=>JSON.parse(execFileSync(process.execPath,['-e','const e=require(process.argv[1]).modelEnv({});process.stdout.write(JSON.stringify({input:e.VODER_MODEL_INPUT||null}));',module],{env:{PATH:process.env.PATH,HOME:home},encoding:'utf8'}));
  const base={VODER_MODEL_ID:'deepseek-flash',VODER_MODEL_API_KEY:'fixture-not-real'};
  await writeFile(config,JSON.stringify(base),{mode:0o600});assert.equal(probe().input,'text,image');
  await writeFile(config,JSON.stringify({...base,VODER_MODEL_INPUT:'text'}));assert.equal(probe().input,'text');
  await writeFile(config,JSON.stringify({...base,VODER_MODEL_ENDPOINT:'https://proxy.invalid/v1'}));assert.equal(probe().input,null);
  await writeFile(config,JSON.stringify({...base,VODER_MODEL_ID:'deepseek-v4-pro'}));assert.equal(probe().input,null);
 }finally{await rm(home,{recursive:true,force:true});}
});
test('frontend and core share exactly the same capacity policy source',()=>{
 const fs=require('node:fs');assert.equal(fs.readFileSync(new URL('../src/input-capacity.cjs',import.meta.url),'utf8'),fs.readFileSync(new URL('../../voder-vscode/src/input-capacity.cjs',import.meta.url),'utf8'));
});
