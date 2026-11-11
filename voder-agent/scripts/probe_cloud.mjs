#!/usr/bin/env node
/** Real Pi session + real voderd state-observation, no paid model requests. */
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, rm, realpath } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
const root=resolve(process.argv[2] ?? '/root/autodl-tmp/voder-agent');
const workspace=resolve(process.argv[3] ?? '/root/autodl-tmp');
const core=join(root,'packages/voder-agent-core');
const bridge=join(root,'packages/voder-native-bridge/src');
const piFile=join(core,'node_modules/@earendil-works/pi-coding-agent/dist/index.js');
const bin=join(workspace,'target/debug/voderd');
let stage, piSession, observer;
function pass(x){console.log(`${x}=PASS`)}
try {
 const pi=await import(pathToFileURL(await realpath(piFile)).href);
 assert.equal(typeof pi.createAgentSession,'function');
 pass('OFFICIAL_PI_LOCAL_ESM_GATE');
 const {PiModelHost}=await import(pathToFileURL(join(core,'src/pi-host.mjs')).href);
 const {NativeVgrObserver}=await import(pathToFileURL(join(bridge,'index.mjs')).href);
 const {NativeVgrCommitVerifier}=await import(pathToFileURL(join(bridge,'commit-verifier.mjs')).href);
 stage=await mkdtemp(join(tmpdir(),'voder-stage2-real-'));
 const repo=join(stage,'repo');await mkdir(repo);
 execFileSync('git',['init','-q',repo]);
 await writeFile(join(repo,'README.md'),'Voder Stage2 isolated native gate\n');
 execFileSync('git',['-C',repo,'add','README.md']);
 execFileSync('git',['-C',repo,'-c','user.name=Voder Test','-c','user.email=voder-test@example.invalid','commit','-qm','baseline']);
 const {createAgentSession,...rest}=pi;
 let rawSession;
 const host=new PiModelHost({upstream:{...rest,createAgentSession:async args=>{
   const opened=await createAgentSession(args); rawSession=opened.session;return opened;
 }}});
 let rejected=0;
 piSession=await host.createSession({cwd:repo,systemPrompt:'Isolated no-model-call audit.',tools:[{
   name:'voder_authorized_action',description:'VGR native authorization is mandatory.',
   async execute(){rejected++;const e=new Error('native mutation disabled in audit');e.code='NO_NATIVE_VGR_AUTHORITY';throw e;}
 }]});
 const inventory=piSession.inspectTools();
 assert.deepEqual(inventory.active,['voder_authorized_action']);
 assert.deepEqual(inventory.callable,['voder_authorized_action']);
 assert.deepEqual(inventory.registered,['voder_authorized_action']);
 pass('REAL_PI_EXACT_TOOL_BOUNDARY_GATE');
 rawSession.setActiveToolsByName(['bash','write','edit','tool_search','mcp__bad','voder_authorized_action']);
 assert.deepEqual(piSession.inspectTools().callable,['voder_authorized_action']);
 const result=await rawSession.getToolDefinition('voder_authorized_action').execute('audit',{
   action:{kind:'command',program:'/bin/sh',args:['-c','exit 0']}
 },new AbortController().signal);
 assert.equal(result.details?.status,'DENIED');assert.equal(rejected,1);
 pass('REAL_PI_UNAUTHORIZED_TOOL_DENIAL_GATE');
 piSession.dispose();piSession=null;
 const stateDir=join(stage,'state');
 observer=await new NativeVgrObserver({binary:bin,stateDir}).start();
 const created=await observer.createSession(repo);
 const status=await observer.state();
 assert.equal(status.session_id,created.session_id);
 assert.deepEqual(status.runs,[]);
 pass('REAL_RUST_DAEMON_SESSION_STATE_GATE');
 observer.close();observer=null;
 const verifier=new NativeVgrCommitVerifier({binary:bin,stateDir});
 const fake={status:'COMMITTED',session_id:created.session_id,action_id:'a-fake',run_id:'r-fake',commit_digest:'a'.repeat(64)};
 await assert.rejects(()=>verifier.confirmCommit(fake,{session_id:created.session_id,action_id:'a-fake'}));
 pass('REAL_RUST_DAEMON_FAKE_COMMIT_DENIAL_GATE');
 console.log('REAL_MODEL_PROVIDER_CALL=NOT_TESTED');
 console.log('MODEL_TO_VGR_MUTATION_CHAIN=NOT_CONNECTED');
 console.log('PRODUCTION_GATE=NOT_PASSED');
 console.log('VODER_STAGE2_REAL_NATIVE_BOUNDARY=PASS');
} catch(e){console.error(`VODER_STAGE2_REAL_NATIVE_BOUNDARY=FAIL code=${e.code??'ERROR'} message=${e.message}`);process.exitCode=1;}
finally{try{piSession?.dispose()}catch{};try{observer?.close()}catch{};if(stage)await rm(stage,{recursive:true,force:true});}
