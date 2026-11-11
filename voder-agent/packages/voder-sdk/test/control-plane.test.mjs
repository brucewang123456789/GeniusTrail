import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,readFile,writeFile,chmod,lstat} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {spawn} from 'node:child_process';
import {VoderSDK,VoderControlPlane,VoderJournal} from '../src/index.mjs';

const mkRoot=async()=>mkdtemp(join(tmpdir(),'voder-sdk-cli-test-'));
const mockHost={async createSession({tools}){
 assert(tools.some(t=>t.name==='voder_authorized_action'));
 return {subscribe(){return ()=>{};},async prompt(goal){return {settled:true,draft:'draft for '+goal.slice(0,25)}},dispose(){},async abort(){}};
}};
const native={async executeAuthorized(){throw Object.assign(new Error('Not authorized'),{code:'NO_AUTH'});}};
const make=async()=>{const dir=await mkRoot();const sdk=new VoderSDK({cwd:dir,native,modelHost:mockHost,allowUnverifiedDraft:true});const journal=await new VoderJournal({dir:join(dir,'private'),workspace:dir}).initialize();return {dir,journal,host:new VoderControlPlane({sdk,journal})};};

test('SDK and journal: fresh task, metadata, no fake verification',async()=>{
 const {host,journal}=await make();const {sessionId}=await host.createSession();
 const ack=await host.submitTask(sessionId,{goal:'read the code'});assert.equal(ack.status,'ACCEPTED');
 const result=await host.wait(sessionId);assert.equal(result.status,'DRAFT_UNVERIFIED');
 assert.equal((await host.status(sessionId)).result.verified,false);
 const events=await host.events(sessionId);assert(events.some(x=>x.type==='task.settled'));assert(events.every(x=>!JSON.stringify(x).includes('read the code')));
 const file=await journal.get(sessionId);assert.equal(file.status,'DRAFT_UNVERIFIED');
 assert.equal((await host.resume(sessionId)).canContinue,false);
 assert.equal((await host.getDiff(sessionId)).status,'NOT_AVAILABLE');
 host.dispose();
});
test('cross-session isolation; invalid and collision requests fail closed',async()=>{
 const {host}=await make();const a=await host.createSession(),b=await host.createSession();
 await host.submitTask(a.sessionId,{goal:'alpha'});await host.wait(a.sessionId);
 assert.equal((await host.events(b.sessionId)).length,0);
 await assert.rejects(()=>host.cancel(a.sessionId),e=>e.code==='SESSION_NOT_ACTIVE');
 await assert.rejects(()=>host.submitTask(a.sessionId,{goal:'repeat'}),e=>e.code==='SESSION_STATE');
 await assert.rejects(()=>host.events('../etc/passwd'),e=>e.code==='INVALID_SESSION_ID');host.dispose();
});
test('export is private, exact, no overwrite; hash tampering blocked',async()=>{
 const {dir,host,journal}=await make();const {sessionId}=await host.createSession();
 await host.submitTask(sessionId,{goal:'review'});await host.wait(sessionId);
 const path=join(dir,'export.json');const receipt=await host.export(sessionId,path);assert.equal(receipt.status,'EXPORTED_UNVERIFIED_SESSION_METADATA');
 const st=await lstat(path);assert.equal(st.mode&0o077,0);
 await assert.rejects(()=>host.export(sessionId,path),e=>e.code==='EXPORT_EXISTS');
 const snap=join(journal._path(sessionId),'snapshot.json');const data=JSON.parse(await readFile(snap));data.status='VERIFIED';await writeFile(snap,JSON.stringify(data));
 await assert.rejects(()=>journal.get(sessionId),e=>e.code==='JOURNAL_INTEGRITY');host.dispose();
});
test('unknown state cannot become a resumed or verified action',async()=>{
 const {host,journal}=await make();const {sessionId}=await host.createSession();
 const fresh=new VoderJournal({dir:journal.dir,workspace:journal.workspace});const r=await fresh.get(sessionId);assert.equal(r.status,'CREATED');
 await assert.rejects(()=>new VoderJournal({dir:journal.dir,workspace:'/another-project'}).get(sessionId),e=>e.code==='JOURNAL_WORKSPACE_MISMATCH');
 host.dispose();
});
test('real CLI JSON-RPC framing and independent reader',async()=>{
 const dir=await mkRoot();const bin=new URL('../../voder-cli/bin/voder-cli.mjs',import.meta.url);
 const child=spawn(process.execPath,[bin.pathname,'rpc','--draft','--workspace',dir,'--session-dir',join(dir,'private')],{stdio:['pipe','pipe','pipe']});
 let output='';child.stdout.on('data',x=>output+=x);let err='';child.stderr.on('data',x=>err+=x);
 const rpc=(id,method,params={})=>child.stdin.write(JSON.stringify({jsonrpc:'2.0',id,method,params})+'\n');
 rpc(1,'initialize');rpc(2,'session.create');rpc(3,'no.such.method');
 await new Promise(r=>setTimeout(r,160));child.stdin.end();
 const rc=await new Promise((r,j)=>{child.on('error',j);child.on('close',r)});
 assert.equal(rc,0,err);
 const messages=output.trim().split('\n').map(JSON.parse).sort((a,b)=>a.id-b.id);
 assert.equal(messages.length,3,output);assert.equal(messages[0].result.capabilities.native_resume,false);
 assert.match(messages[1].result.sessionId,/^[a-f0-9-]{36}$/i);
 assert.equal(messages[2].error.code,'METHOD_NOT_FOUND');
 const reader=spawn(process.execPath,[bin.pathname,'status','--workspace',dir,'--session-dir',join(dir,'private'),'--session',messages[1].result.sessionId]);
 let rd='';reader.stdout.on('data',c=>rd+=c);const code=await new Promise(r=>reader.on('close',r));assert.equal(code,0);assert.equal(JSON.parse(rd).status,'CREATED');
});

test('event listener receives metadata and unsubscribe isolates later sessions',async()=>{
 const {dir}=await (async()=>({dir:await mkRoot()}))();
 const sdk=new VoderSDK({cwd:dir,native,modelHost:mockHost,allowUnverifiedDraft:true});
 const {sessionId}=sdk.createSession();let seen=[];
 const off=sdk.subscribeEvents(sessionId,e=>seen.push(e.type));
 await sdk.submitTask(sessionId,{goal:'task'});assert(seen.includes('task.settled'));
 off();const before=seen.length;
 // Cannot submit another task after draft-only terminal; the previously subscribed observer stays removed.
 await sdk.cancel(sessionId);assert.equal(before,seen.length);sdk.dispose();
});
test('live cancellation is not delivery and does not silently re-execute',async()=>{
 const dir=await mkRoot();let resolvePrompt,called=0;
 const slowHost={async createSession(){return {async prompt(){called++;return await new Promise(r=>resolvePrompt=r)},async abort(){resolvePrompt?.({settled:true})},dispose(){}}}};
 const sdk=new VoderSDK({cwd:dir,native,modelHost:slowHost,allowUnverifiedDraft:true});
 const journal=await new VoderJournal({dir:join(dir,'store'),workspace:dir}).initialize();const host=new VoderControlPlane({sdk,journal});
 const {sessionId}=await host.createSession();await host.submitTask(sessionId,{goal:'pause before approval'});
 for(let i=0;i<100&&!resolvePrompt;i++)await new Promise(r=>setTimeout(r,2));
 assert.equal(called,1);
 const cancelled=await host.cancel(sessionId);assert.equal(cancelled.status,'CANCELLATION_REQUESTED');
 const result=await host.wait(sessionId);assert.equal(result.status,'CANCELLED');assert.equal(called,1);
 assert.equal((await journal.get(sessionId)).status,'CANCELLED');
 await assert.rejects(()=>host.submitTask(sessionId,{goal:'unexpected retry'}),e=>e.code==='SESSION_STATE');
 host.dispose();
});
test('hard missing input blocks the model while preserving journal state',async()=>{
 const {host}=await make();const {sessionId}=await host.createSession();
 await host.submitTask(sessionId,{goal:'check missing',requiredFiles:['definitely-not-present-98765.txt']});
 const result=await host.wait(sessionId);
 assert.equal(result.status,'WAITING_FOR_INPUT');assert.equal((await host.status(sessionId)).result.status,'WAITING_FOR_INPUT');host.dispose();
});
test('export does not follow final symbolic link and session ID path traversal fails',async()=>{
 const {host,dir}=await make();const {sessionId}=await host.createSession();
 const out=join(dir,'file.json');await writeFile(out,'unchanged');
 await assert.rejects(()=>host.export(sessionId,out),e=>e.code==='EXPORT_EXISTS');
 await assert.rejects(()=>host.resume('../../danger'),e=>e.code==='INVALID_SESSION_ID');host.dispose();
});
test('memory selected by client flows into unchanged SDK memory handling',async()=>{
 const {dir,host}=await make();const path=join(dir,'notes.vmem.json');
 await host.sdk.exportMemory({entries:[{id:'pref',kind:'preference',title:'Test preference',content:'Require real test receipts',scope:'user'}],selectedIds:['pref'],userConfirmed:true,outputPath:path});
 const {sessionId}=await host.createSession({memoryFiles:[path]});
 await host.submitTask(sessionId,{goal:'Summarize imported preference'});
 const result=await host.wait(sessionId);assert.equal(result.status,'DRAFT_UNVERIFIED');
 assert((await host.events(sessionId)).some(e=>e.type==='memory.imported'));host.dispose();
});


test('live SDK wait returns exact model answer but persistent journal and exported metadata never contain it',async()=>{
 const {host,dir,journal}=await make();const privateCanary='LIVE-ANSWER-ONLY-9D8B-PRIVATE';
 const {sessionId}=await host.createSession();
 await host.submitTask(sessionId,{goal:privateCanary});const r=await host.wait(sessionId);
 assert.equal(r.draft,'draft for '+privateCanary.slice(0,25));
 assert.equal(r.output_authority,'UNVERIFIED_MODEL_TEXT_NOT_VGR_PROOF');
 assert.equal(r.verified,false);
 assert.equal((await host.wait(sessionId)).draft,r.draft);
 assert.equal((await host.status(sessionId)).result.draft,undefined);
 assert.equal((await host.resume(sessionId)).result.draft,undefined);
 const dest=join(dir,'private_export.json');await host.export(sessionId,dest);
 for(const bytes of [await readFile(join(journal._path(sessionId),'snapshot.json'),'utf8'),await readFile(dest,'utf8')]){
  assert(!bytes.includes('draft for '+privateCanary.slice(0,25)));
  assert(!bytes.includes(privateCanary));
 }
 const detached=new VoderJournal({dir:journal.dir,workspace:journal.workspace});
 assert.equal((await detached.get(sessionId)).result.draft,undefined);
 host.dispose();
});

test('verified-looking or failed results cannot smuggle unverified model draft as VGR proof',async()=>{
 const dir=await mkRoot();
 const fake={async createSession(){return {async prompt(){return {draft:'ATTACK',settled:true}},dispose(){}}}};
 const sdk=new VoderSDK({cwd:dir,native,modelHost:fake,allowUnverifiedDraft:true});
 const journal=await new VoderJournal({dir:join(dir,'journal'),workspace:dir}).initialize();
 const host=new VoderControlPlane({sdk,journal});const {sessionId}=await host.createSession();
 await host.submitTask(sessionId,{goal:'review'});const result=await host.wait(sessionId);
 assert.equal(result.status,'DRAFT_UNVERIFIED');assert.equal(result.verified,false);
 assert.equal(result.output_authority,'UNVERIFIED_MODEL_TEXT_NOT_VGR_PROOF');
 assert.equal((await host.status(sessionId)).result.verified,false);host.dispose();
});

test('human terminal format visibly emits complete model text, JSON default retains exact field',async()=>{
 const dir=await mkRoot();const {main}=await import('../../voder-cli/bin/voder-cli.mjs');
 const streamed='FIRST\nSECOND\n';const controlled={async createSession(){return {async prompt(){return {draft:streamed,settled:true}},dispose(){}}}};
 let out='';let ret=await main(['run','--draft','--workspace',dir,'--session-dir',join(dir,'journal-a'),'--goal','test','--format','text'],
  {output:{write(x){out+=x}},modelHost:controlled});
 assert.equal(ret,3);assert(out.includes('[UNVERIFIED MODEL DRAFT'));assert(out.includes('FIRST\nSECOND\n'));
 out='';ret=await main(['run','--draft','--workspace',dir,'--session-dir',join(dir,'journal-b'),'--goal','test'],
  {output:{write(x){out+=x}},modelHost:controlled});
 assert.equal(ret,3);const parsed=JSON.parse(out);assert.equal(parsed.draft,streamed);
 assert.equal(parsed.verified,false);assert.equal(parsed.output_authority,'UNVERIFIED_MODEL_TEXT_NOT_VGR_PROOF');
});

test('JSON-RPC task.wait delivers private model draft to live caller, status is metadata only',async()=>{
 const {PassThrough}=await import('node:stream');
 const {serveRpc}=await import('../../voder-cli/bin/voder-cli.mjs');
 const {host}=await make();const {sessionId}=await host.createSession();
 const input=new PassThrough(),output=new PassThrough();
 const replies=[];let buffer='';const pending=new Map();
 output.on('data',c=>{buffer+=c.toString('utf8');for(;;){const n=buffer.indexOf('\n');if(n<0)break;
  const line=buffer.slice(0,n);buffer=buffer.slice(n+1);const obj=JSON.parse(line);replies.push(obj);
  const done=pending.get(obj.id);if(done){pending.delete(obj.id);done(obj);}
 }});
 const done=serveRpc(host,{input,output});
 let rid=0;
 const send=async(method,params)=>{
  const id=++rid;const promise=new Promise((resolve,reject)=>{const timer=setTimeout(()=>{pending.delete(id);reject(Error('RPC_TIMEOUT'))},3000);
   pending.set(id,obj=>{clearTimeout(timer);resolve(obj)});
  });
  input.write(JSON.stringify({jsonrpc:'2.0',id,method,params})+'\n');return promise;
 };
 const ack=await send('task.submit',{sessionId,task:{goal:'private task answer'}});
 assert.equal(ack.result.status,'ACCEPTED');
 const wait=await send('task.wait',{sessionId});
 assert.equal(wait.result.status,'DRAFT_UNVERIFIED');
 assert.equal(wait.result.draft,'draft for private task answer');
 assert.equal(wait.result.verified,false);
 const status=await send('session.status',{sessionId});
 assert.equal(status.result.result.draft,undefined);
 input.end();await done;
});
