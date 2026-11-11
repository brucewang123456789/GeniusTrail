import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {AgentKernel,VoderError,assembleSystemPrompt} from '../src/index.mjs';
import {VoderSDK} from '../../voder-sdk/src/index.mjs';
async function tmp(fn){const d=await mkdtemp(join(tmpdir(),'voder-context-interface-'));try{return await fn(d)}finally{await rm(d,{recursive:true,force:true})}}

test('trusted context supplement is composed exactly once per new model session, never per tool step',async()=>tmp(async cwd=>{
 let composed=0,received=[],calls=0;
 const kernel=new AgentKernel({cwd,sessionId:'ctx-1',broker:{},
   contextProvider:async args=>{composed++;assert.equal(args.stage,'next');assert.equal(args.task,'repair');assert.ok(Object.isFrozen(args));return 'SKILL_OVERLAY_FROM_TRUSTED_HOST';},
   modelHost:{async createSession(options){received.push(options.systemPrompt);return {async prompt(){calls++;},dispose(){}}}}});
 const r=await kernel.submit({goal:'repair'});
 assert.equal(r.status,'BLOCKED');assert.equal(composed,1);assert.equal(calls,1);
 assert.match(received[0],/01 Current Request Supremacy/);
 assert.match(received[0],/14 Separate Core Discipline/);
 assert.match(received[0],/\[VODER_TRUSTED_CONTEXT_SUPPLEMENT\]\nSKILL_OVERLAY_FROM_TRUSTED_HOST/);
 kernel.close();
}));

test('missing inputs stop before optional context composition',async()=>tmp(async cwd=>{
 let called=0;
 const kernel=new AgentKernel({cwd,broker:{},contextProvider:()=>{called++;return 'x'},
   modelHost:{async createSession(){throw Error('must not start')}}});
 assert.equal((await kernel.submit({goal:'edit',requiredFiles:['missing.txt']})).status,'WAITING_FOR_INPUT');
 assert.equal(called,0);kernel.close();
}));

test('invalid or excessive context fails closed before creating model session',async()=>tmp(async cwd=>{
 for(const [provider,code] of [[()=>({systemPrompt:'arbitrary'}),'INVALID_CONTEXT'],[()=>('X'.repeat(49000)),'CONTEXT_OVERFLOW']]){
   let modelCalls=0;
   const kernel=new AgentKernel({cwd,broker:{},contextProvider:provider,modelHost:{async createSession(){modelCalls++;}}});
   await assert.rejects(kernel.submit({goal:'repair'}),{code});assert.equal(modelCalls,0);kernel.close();
 }
 assert.throws(()=>new AgentKernel({cwd,broker:{},contextProvider:'untrusted',modelHost:{createSession(){}}}),{code:'INVALID_CONTEXT_PROVIDER'});
}));

test('no provider retains exact original system prompt',async()=>tmp(async cwd=>{
 let prompt;
 const kernel=new AgentKernel({cwd,broker:{},modelHost:{async createSession({systemPrompt}){prompt=systemPrompt;return {async prompt(){},dispose(){}}}}});
 await kernel.submit({goal:'baseline'});
 assert.equal(prompt,assembleSystemPrompt('next',{task:'baseline'}));kernel.close();
}));

test('SDK exposes trusted context seam without altering core tool policy or VGR status',async()=>{
 const d=await mkdtemp(join(tmpdir(),'voder-sdk-ctx-'));
 try{
   let prompt,calls=0;
   const sdk=new VoderSDK({cwd:d,native:{async executeAuthorized(){throw Error('not called')}},
     contextProvider:async ({stage,task})=>{assert.equal(stage,'next');assert.equal(task,'review');calls++;return 'TRUSTED_MD_EXTENSION';},
     modelHost:{async createSession(o){prompt=o.systemPrompt;return {async prompt(){},dispose(){}}}}});
   const {sessionId}=sdk.createSession({id:'ctx-sdk'});
   assert.equal((await sdk.submitTask(sessionId,{goal:'review'})).status,'BLOCKED');
   assert.equal(calls,1);assert.match(prompt,/TRUSTED_MD_EXTENSION/);assert.match(prompt,/14 Separate Core Discipline/);
   sdk.dispose();
   assert.throws(()=>new VoderSDK({cwd:d,contextProvider:'not a function'}),{code:'INVALID_CONTEXT_PROVIDER'});
 }finally{await rm(d,{force:true,recursive:true})}
});
