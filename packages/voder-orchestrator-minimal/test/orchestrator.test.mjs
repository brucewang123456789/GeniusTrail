import {fileURLToPath} from 'node:url';
import test from 'node:test';import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,symlink,chmod,lstat} from 'node:fs/promises';import {tmpdir} from 'node:os';import {join,resolve} from 'node:path';
import {sha} from '../lib/safety.mjs';import {Journal} from '../lib/journal.mjs';import {parseProposal} from '../lib/patch.mjs';import {configCheck,runWorkflow} from '../lib/engine.mjs';
const voderRoot=process.env.VODER_ROOT_TEST || process.env.VODER_ROOT || fileURLToPath(new URL('../../../', import.meta.url));
async function fixture(){const home=await mkdtemp(join(tmpdir(),'voder-orch-test-'));const source=join(home,'source'),state=join(home,'state');await mkdir(source);await mkdir(state,{mode:0o700});await mkdir(join(source,'src'));await writeFile(join(source,'src/main.txt'),'BUG\n');await writeFile(join(source,'requirements.txt'),'safe\n');
 const config={version:1,project:source,stateRoot:state,goal:'Fix BUG',requiredFiles:['src/main.txt'],allowedPaths:['src/main.txt'],validators:[{name:'check',program:'/usr/bin/python3',args:['-m','unittest'],timeoutMs:5000}],maxSteps:3,maxModelCalls:3};return {home,source,state,config};}
function provider({text,verify,failOnModel=false}){let modelCalls=0,sandboxCalls=0,prompts=[];return {get modelCalls(){return modelCalls},get sandboxCalls(){return sandboxCalls},get prompts(){return prompts},async model(_run,_i,prompt){modelCalls++;prompts.push(prompt);if(failOnModel)throw Error('UPSTREAM_ERROR');return {text:typeof text==='function'?text(modelCalls,prompt):text,usage:{totalTokens:10}}},async sandbox(_dir,_i,workspace){sandboxCalls++;if(verify==='transport')throw Error('MODAL_TRANSPORT_FAILED');const s=await readFile(join(workspace,'src/main.txt'),'utf8');const ok=verify?verify(s,sandboxCalls):s==='FIXED\n';return {passed:ok,stdout:ok?'ok':'FAIL BUG',stderr:'trace',receipt:{verified:true}};}};}
const fix={version:1,patches:[{path:'src/main.txt',action:'replace',beforeSha256:sha('BUG\n'),content:'FIXED\n'}]};
test('real loop: baseline fails, model repairs isolated worktree, validator passes, original untouched',async()=>{const f=await fixture();const a=provider({text:JSON.stringify(fix)});const r=await runWorkflow(f.config,{adapters:a,voderRoot});assert.equal(r.status,'VERIFIED_STAGED_VGR_PENDING');assert.equal(r.rank,1);assert.equal(r.deliveryAllowed,false);assert.equal(r.authority,'NATIVE_VGR_NOT_CONNECTED');assert.equal(await readFile(join(f.source,'src/main.txt'),'utf8'),'BUG\n');assert.equal(await readFile(join(r.workspace,'src/main.txt'),'utf8'),'FIXED\n');assert.equal(a.modelCalls,1);assert.equal(a.sandboxCalls,2);const j=new Journal(join(r.runDir,'events.jsonl'));assert.ok((await j.verify()).some(e=>e.event==='stage.verified'));});
test('no evidence advance = rollback; no false delivery',async()=>{const f=await fixture();const a=provider({text:JSON.stringify(fix),verify:()=>false});const r=await runWorkflow(f.config,{adapters:a,voderRoot});assert.equal(r.status,'BLOCKED');assert.equal(await readFile(join(r.workspace,'src/main.txt'),'utf8'),'BUG\n');assert.equal(a.modelCalls,2);});
test('fail closed on missing referenced file without calling model or sandbox',async()=>{const f=await fixture();f.config.requiredFiles=['missing.py'];const a=provider({text:'x'});await assert.rejects(()=>runWorkflow(f.config,{adapters:a,voderRoot}),/MISSING_REQUIRED_FILE/);assert.equal(a.modelCalls,0);assert.equal(a.sandboxCalls,0);});
test('patch cannot modify outside trusted scope or path traversal',()=>{assert.throws(()=>parseProposal(JSON.stringify({version:1,patches:[{path:'../secrets',action:'create',content:'A'}]}),new Set(['src/main.txt'])),/UNSAFE_RELATIVE_PATH/);assert.throws(()=>parseProposal(JSON.stringify({version:1,patches:[{path:'src/other',action:'create',content:'A'}]}),new Set(['src/main.txt'])),/PATCH_OUTSIDE_ALLOWED_SCOPE/);});
test('preimage mismatch produces explicit rejected patch, no mutation',async()=>{const f=await fixture();const wrong=structuredClone(fix);wrong.patches[0].beforeSha256='0'.repeat(64);const r=await runWorkflow(f.config,{adapters:provider({text:JSON.stringify(wrong)}),voderRoot});assert.equal(r.status,'BLOCKED');assert.equal(await readFile(join(r.workspace,'src/main.txt'),'utf8'),'BUG\n');});
test('validator transport failure triggers rollback and blocks delivery',async()=>{const f=await fixture();const a=provider({text:JSON.stringify(fix),verify:(s,n)=>{if(n>=2)throw Error('bad')}});a.sandbox=async(_d,n,workspace)=>{if(n>1)throw Error('MODAL_DISCONNECTED');return {passed:false,stdout:'fail',stderr:'',receipt:{}};};const r=await runWorkflow(f.config,{adapters:a,voderRoot});assert.equal(r.status,'BLOCKED_UNCERTAIN');assert.equal(await readFile(join(r.workspace,'src/main.txt'),'utf8'),'BUG\n');});
test('upstream model failure leaves original and staged files unchanged',async()=>{const f=await fixture();const r=await runWorkflow(f.config,{adapters:provider({text:'',failOnModel:true}),voderRoot});assert.equal(r.status,'BLOCKED_UNCERTAIN');assert.equal(await readFile(join(r.workspace,'src/main.txt'),'utf8'),'BUG\n');});
test('no shell commands or tool access from model proposal',()=>{assert.throws(()=>parseProposal(JSON.stringify({version:1,patches:[fix.patches[0]],command:'rm -rf /'}),new Set(['src/main.txt'])),/PATCH_UNKNOWN_FIELDS/);});
test('reject symlink in source before model call',async()=>{const f=await fixture();await symlink('/etc/passwd',join(f.source,'bad_link'));const a=provider({text:JSON.stringify(fix)});await assert.rejects(()=>runWorkflow(f.config,{adapters:a,voderRoot}),/SYMLINK_BLOCKED/);assert.equal(a.modelCalls,0);});
test('state dir must be private and separated from project',async()=>{const f=await fixture();await chmod(f.state,0o755);await assert.rejects(()=>runWorkflow(f.config,{adapters:provider({text:'x'}),voderRoot}),/STATE_DIRECTORY_NOT_PRIVATE/);});
test('malformed or duplicated model response cannot authorize anything',async()=>{const f=await fixture();const a=provider({text:'```json\n{}\n```'});const r=await runWorkflow(f.config,{adapters:a,voderRoot});assert.equal(r.status,'BLOCKED');assert.equal(await readFile(join(r.workspace,'src/main.txt'),'utf8'),'BUG\n');});
test('formal profile imports only user-declared memory and verifies staging',async()=>{const f=await fixture();const memory=join(f.home,'memory.md');await writeFile(memory,'PRIORITY: root cause first');f.config.memoryFiles=[memory];const a=provider({text:JSON.stringify(fix)});const r=await runWorkflow(f.config,{formal:true,adapters:a,voderRoot});assert.equal(r.status,'VERIFIED_STAGED_VGR_PENDING');assert.match(a.prompts[0],/PRIORITY: root cause first/);});
test('minimal profile rejects formal memory and excessive steps',async()=>{const f=await fixture();f.config.memoryFiles=['/tmp/test'];assert.throws(()=>configCheck(f.config,false),/MEMORY_FORMAL_ONLY/);delete f.config.memoryFiles;f.config.maxSteps=6;assert.throws(()=>configCheck(f.config,false),/MINIMAL_STEPS_LIMIT/);});
test('journal independently detects tampered evidence',async()=>{const f=await fixture();const path=join(f.state,'journal.jsonl');const j=new Journal(path);await j.write('test',{rank:1});await j.verify();let raw=await readFile(path,'utf8');await writeFile(path,raw.replace('"rank":1','"rank":2'));await assert.rejects(()=>new Journal(path).verify(),/JOURNAL_TAMPERED/);});
test('existing destination remains unmodified even when staging succeeds',async()=>{const f=await fixture();const initial=sha(await readFile(join(f.source,'src/main.txt')));const r=await runWorkflow(f.config,{adapters:provider({text:JSON.stringify(fix)}),voderRoot});assert.equal(sha(await readFile(join(f.source,'src/main.txt'))),initial);assert.equal((await lstat(join(f.source,'src/main.txt'))).isFile(),true);});

// Single-writer crash-safety regressions (no provider or Modal credentials).
test('single writer lease refuses concurrent work and preserves the incumbent',async()=>{
 const f=await fixture();
 const {withRunLease}=await import('../lib/lease.mjs');
 let release;const wait=new Promise(resolve=>{release=resolve});
 let ready;const entered=new Promise(resolve=>{ready=resolve});
 const first=withRunLease(f.state,async()=>{ready();await wait;return 'FIRST';});
 await entered;
 await assert.rejects(withRunLease(f.state,async()=>{throw Error('SECOND_EXECUTED')}),/RECOVERY_REQUIRED_EXISTING_RUN_LEASE/);
 release();assert.equal(await first,'FIRST');
 assert.equal(await withRunLease(f.state,async()=>42),42);
});
test('crash residue demands recovery, never steals the lock',async()=>{
 const f=await fixture();
 const {withRunLease}=await import('../lib/lease.mjs');
 const lock=join(f.state,'.voder-orchestrator.lock');
 await writeFile(lock,'{\"version\":1,\"nonce\":\"crashed\"}',{mode:0o600});
 await assert.rejects(withRunLease(f.state,async()=>false),/RECOVERY_REQUIRED_EXISTING_RUN_LEASE/);
 assert.match(await readFile(lock,'utf8'),/crashed/);
});
test('lease tamper detection never reports success',async()=>{
 const f=await fixture();
 const {withRunLease}=await import('../lib/lease.mjs');
 const lock=join(f.state,'.voder-orchestrator.lock');
 await assert.rejects(withRunLease(f.state,async()=>{await writeFile(lock,'{\"nonce\":\"attacker\"}');}),/RUN_LEASE_TAMPERED/);
});
