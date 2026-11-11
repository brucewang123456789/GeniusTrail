import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdtemp,mkdir,writeFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {AgentKernel} from '../src/kernel.mjs';
import {PiModelHost} from '../src/pi-host.mjs';
import {VgrBroker} from '../src/broker.mjs';
const digest=s=>createHash('sha256').update(s).digest('hex');
async function fixture(fn){const cwd=await mkdtemp(join(tmpdir(),'voder-orchestrator-'));try{await mkdir(join(cwd,'src'));await writeFile(join(cwd,'src','sample.py'),'print(0)\n');return await fn(cwd);}finally{await rm(cwd,{recursive:true,force:true})}}
function fakePi(onPrompt){
  return {
    getAgentDir:()=>'/trusted',
    SessionManager:{inMemory:()=>({})},
    DefaultResourceLoader:class{constructor(o){this.options=o}async reload(){}},
    defineTool:x=>x,
    createAgentSession:async opts=>{
      const defs=opts.customTools;
      const names=defs.map(x=>x.name);
      return {session:{
        getActiveToolNames:()=>names,
        getCallableToolNames:()=>names,
        getAllTools:()=>names.map(name=>({name,exposure:'direct'})),
        async prompt(goal){await onPrompt(defs,goal,opts)},
        dispose(){},subscribe(){return()=>{}},abort(){}
      }};
    }
  };
}
function receiptFor(a){return {authority:'VODER_NATIVE_VGR',session_id:a.session_id,action_id:a.action_id,run_id:`run-${a.action_id}`,status:'COMMITTED',journal_digest:'a'.repeat(64),commit_digest:'b'.repeat(64)};}
function nativeBroker(cwd,sessionId,onNative){return new VgrBroker({sessionId,workspaceRoot:cwd,native:{async executeAuthorized(a){return onNative(a)}},commitVerifier:{async confirmCommit(r,a){return {verified:true,authority:'VODER_NATIVE_VGR_STATE',session_id:a.session_id,action_id:a.action_id,run_id:r.run_id,commit_digest:r.commit_digest}}}});}
const candidate={relative_path:'src/sample.py',old_sha256:digest('print(0)\n'),new_content:'print(2)\n'};
const signal=new AbortController().signal;
async function call(defs,name,args){return defs.find(x=>x.name===name).execute('t',args,signal)}
const parsed=r=>JSON.parse(r.content[0].text);
test('model cannot bypass a required real sandbox PASS to commit',async()=>fixture(async cwd=>{
 let nativeCount=0, modelOutcome;
 const host=new PiModelHost({upstream:fakePi(async defs=>{
   modelOutcome=parsed(await call(defs,'voder_authorized_action',{action:{kind:'patch',...candidate}}));
 })});
 const broker=nativeBroker(cwd,'task-1',a=>{nativeCount++;return receiptFor(a)});
 const kernel=new AgentKernel({cwd,modelHost:host,broker,sessionId:'task-1',candidateVerifier:{async verifyCandidate(){throw Error('should not run')}}});
 const result=await kernel.submit({goal:'repair bug'});
 assert.equal(result.status,'BLOCKED');assert.equal(nativeCount,0);assert.equal(modelOutcome.error,'VERIFY_BEFORE_COMMIT');
 assert.equal(result.orchestration.attempted_native_actions,0);
 kernel.close();
}));
test('real diagnostic -> changed candidate -> sandbox PASS -> independent Rust-confirmed COMMIT',async()=>fixture(async cwd=>{
 const observed=[];let verifyCount=0,nativeCount=0;
 const verifier={async verifyCandidate(c,ctx){verifyCount++;assert.equal(ctx.cwd,cwd);assert.equal(c.old_sha256,candidate.old_sha256);return verifyCount===1
  ? {status:'FAIL',backend:'isolated-test-double',diagnostic:'SyntaxError: invalid syntax at line 1'}
  : {status:'PASS',backend:'isolated-test-double',sandboxed:true,evidence_digest:'c'.repeat(64),checks:[{name:'targeted-test',passed:true},{name:'regression',passed:true}]};}};
 const host=new PiModelHost({upstream:fakePi(async(defs,goal,opts)=>{
   assert.match(opts.resourceLoader.options.systemPromptOverride(),/VODER_SANDBOX_CANDIDATE_VERIFICATION/);
   const bad={...candidate,new_content:'print(2\n'};
   observed.push(parsed(await call(defs,'voder_verify_candidate',{candidate:bad})));
   observed.push(parsed(await call(defs,'voder_verify_candidate',{candidate:bad})));
   observed.push(parsed(await call(defs,'voder_verify_candidate',{candidate})));
   observed.push(parsed(await call(defs,'voder_authorized_action',{action:{kind:'patch',...candidate}})));
 })});
 const broker=nativeBroker(cwd,'task-2',a=>{nativeCount++;assert.equal(a.new_content,candidate.new_content);return receiptFor(a)});
 const kernel=new AgentKernel({cwd,modelHost:host,broker,sessionId:'task-2',candidateVerifier:verifier});
 const result=await kernel.submit({goal:'fix SyntaxError then test'});
 assert.equal(result.status,'VERIFIED');assert.equal(verifyCount,2);assert.equal(nativeCount,1);
 assert.equal(observed[0].voder_tool_result.status,'FAIL');
 assert.equal(observed[1].error,'DUPLICATE_CANDIDATE');
 assert.equal(observed[2].voder_tool_result.status,'PASS');
 assert.equal(observed[3].voder_tool_result.status,'COMMITTED');
 assert.equal(result.orchestration.verification_calls,2);
 kernel.close();
}));
test('verifier cannot approve stale initial file digest or malicious path',async()=>fixture(async cwd=>{
 let called=0;
 const host=new PiModelHost({upstream:fakePi(async defs=>{
   const r=parsed(await call(defs,'voder_verify_candidate',{candidate:{...candidate,old_sha256:'f'.repeat(64)}}));
   assert.equal(r.error,'STALE_SOURCE');
 })});
 const kernel=new AgentKernel({cwd,modelHost:host,sessionId:'task-3',broker:{},candidateVerifier:{async verifyCandidate(){called++}}});
 const result=await kernel.submit({goal:'test'});assert.equal(result.status,'BLOCKED');assert.equal(called,0);kernel.close();
}));
test('new model session does not inherit previous task candidate PASS',async()=>fixture(async cwd=>{
 let commits=0;
 const checker={async verifyCandidate(){return {status:'PASS',backend:'isolated-test-double',sandboxed:true,evidence_digest:'a'.repeat(64),checks:[{name:'test',passed:true}]}}};
 const firstHost=new PiModelHost({upstream:fakePi(async defs=>{
   const receipt=parsed(await call(defs,'voder_verify_candidate',{candidate}));
   assert.equal(receipt.voder_tool_result.native_commit.confirmed,true);
 })});
 const firstKernel=new AgentKernel({cwd,modelHost:firstHost,sessionId:'task-4',broker:nativeBroker(cwd,'task-4',a=>{commits++;return receiptFor(a)}),candidateVerifier:checker});
 assert.equal((await firstKernel.submit({goal:'check'})).status,'VERIFIED');firstKernel.close();
 const nextHost=new PiModelHost({upstream:fakePi(async defs=>{
   const res=parsed(await call(defs,'voder_authorized_action',{action:{kind:'patch',...candidate}}));
   assert.equal(res.error,'VERIFY_BEFORE_COMMIT');
 })});
 const nextKernel=new AgentKernel({cwd,modelHost:nextHost,sessionId:'task-4-next',broker:nativeBroker(cwd,'task-4-next',a=>{commits++;return receiptFor(a)}),candidateVerifier:checker});
 const result=await nextKernel.submit({goal:'new task'});
 assert.equal(result.status,'BLOCKED');assert.equal(commits,1);nextKernel.close();
}));
test('source changed after a passing sandbox check blocks privileged commit',async()=>fixture(async cwd=>{
 let nativeCount=0, finalError;
 const host=new PiModelHost({upstream:fakePi(async defs=>{
   const p=parsed(await call(defs,'voder_verify_candidate',{candidate}));assert.equal(p.voder_tool_result.status,'PASS');
   finalError=p.voder_tool_result.native_commit.code;
 })});
 const kernel=new AgentKernel({cwd,modelHost:host,sessionId:'task-source',broker:nativeBroker(cwd,'task-source',a=>{nativeCount++;return receiptFor(a)}),
   candidateVerifier:{async verifyCandidate(){await writeFile(join(cwd,'src/sample.py'),'print(17)\n');return {status:'PASS',backend:'isolated-mock',sandboxed:true,evidence_digest:'a'.repeat(64),checks:[{name:'compile',passed:true}]}}}});
 assert.equal((await kernel.submit({goal:'repair'})).status,'BLOCKED');
 assert.equal(finalError,'STALE_SOURCE');assert.equal(nativeCount,0);kernel.close();
}));
test('sandbox verifier may be declared only via explicit host configuration',async()=>fixture(async cwd=>{
 let available=[];
 const host=new PiModelHost({upstream:fakePi(async defs=>{available=defs.map(x=>x.name)})});
 const kernel=new AgentKernel({cwd,modelHost:host,broker:{},sessionId:'t1'});
 await kernel.submit({goal:'review'});
 assert.equal(available.includes('voder_verify_candidate'),false);
 assert.deepEqual(available,['voder_authorized_action','voder_read_file','voder_list_files','voder_search_code']);
 kernel.close();
}));
