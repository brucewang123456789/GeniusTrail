#!/usr/bin/env node
/** Actual SDK process SIGKILL after real native VGR commit but before SDK received it.
    Only a temporary Git repository is mutated. Modal fees may apply; NO model API.
    This proves no automatic replay of an already-committed action, not rollback replanning. */
import {spawn,execFileSync} from 'node:child_process';
import {randomUUID,createHash} from 'node:crypto';
import {mkdtemp,mkdir,readFile,writeFile,realpath,lstat,chmod} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {VoderSDK,VoderControlPlane,VoderJournal,NativeSdkContinuation} from '../packages/voder-sdk/src/index.mjs';
const root=resolve(process.env.VODER_ROOT||'/root/autodl-tmp');
const node=await realpath(process.env.VODER_NODE_BINARY||join(root,'.voder-tools/node/bin/node'));
const modal=join(root,'packages/voder-sandbox-modal/src/cli.ts');
const sha=v=>createHash('sha256').update(v).digest('hex');
const marker='VODER_SDK_SAFE_RECOVERY_POSTCOMMIT';
const task={goal:'Replace src/input.txt with approved acceptance marker under native VGR control',maxActions:1};
const git=(cwd,...args)=>execFileSync('/usr/bin/git',args,{cwd,encoding:'utf8',timeout:20000,env:{...process.env,GIT_CONFIG_NOSYSTEM:'1',GIT_CONFIG_GLOBAL:'/dev/null'}}).trim();
const checked=async p=>{const s=await lstat(p);if(!s.isFile()||s.isSymbolicLink())throw Error('UNSAFE_BINARY_OR_PLUGIN');return p;};
async function runChild(){
 const dir=process.argv[3],cfg=JSON.parse(await readFile(join(dir,'host-config.json'),'utf8'));
 const cwd=join(dir,'repo');const before='SDK_NATIVE_BEFORE\n',after=marker+'\n';
 const sourceHash=sha(before);
 const modelHost={async createSession({tools}){
  const native=tools.find(x=>x.name==='voder_authorized_action');if(!native)throw Error('MISSING_NATIVE_TOOL');
  return {async prompt(){return await native.execute({kind:'patch',relative_path:'src/input.txt',old_sha256:sourceHash,new_content:after});},dispose(){},async abort(){}};
 }};
 const sdk=new VoderSDK({cwd,nativeGateway:cfg,modelHost});
 const underlying=sdk.native.executeAuthorized.bind(sdk.native);
 sdk.native.executeAuthorized=async(input,options)=>{
  const receipt=await underlying(input,options);
  if(receipt.status!=='COMMITTED')throw Error('UNEXPECTED_NATIVE_STATUS');
  // Parent cannot assume this marker proves commitment; it separately invokes Rust verifier.
  await writeFile(join(dir,'native-committed.marker'),JSON.stringify({session_id:input.session_id,action_id:input.action_id}));
  process.kill(process.pid,'SIGKILL');
  await new Promise(()=>{});
 };
 const journal=await new VoderJournal({dir:join(dir,'journal'),workspace:cwd}).initialize();
 const ctrl=new VoderControlPlane({sdk,journal});const {sessionId}=await ctrl.createSession({autoContinue:true});
 await writeFile(join(dir,'session-id'),sessionId);
 await ctrl.submitTask(sessionId,task);
 setInterval(()=>{},1000);
}
async function parent(){
 if(process.env.VODER_ALLOW_MODAL_CHARGES!=='YES'||process.env.VODER_ENABLE_SDK_SIGKILL!=='YES')throw Error('EXPLICIT_MODAL_AND_SDK_SIGKILL_CONSENT_REQUIRED');
 await checked(join(root,'.voder-build/release/voder-gateway'));await checked(join(root,'.voder-build/release/voderd'));
 await checked(node);await checked(modal);
 const dir=await mkdtemp(join(root,'repair/.voder-sdk-real-crash-'));await chmod(dir,0o700);
 const repo=join(dir,'repo');await mkdir(join(repo,'src'),{recursive:true});
 await writeFile(join(repo,'src/input.txt'),'SDK_NATIVE_BEFORE\n');git(repo,'init','-q');git(repo,'add','src/input.txt');
 git(repo,'-c','user.name=Voder SDK Test','-c','user.email=acceptance@example.invalid','commit','-qm','baseline');
 const args=['-Fx',marker,'src/input.txt'],program='/usr/bin/grep';
 const validatorSha=sha(`${program}\0v1\0[${args.map(v=>JSON.stringify(v)).join(', ')}]`);
 const policyFile=join(dir,'policy.json');await writeFile(policyFile,JSON.stringify({schema:'voder.gateway.policy.v1',repository_path:repo,target_path:'src/input.txt',validator_program:program,validator_args:args,validator_sha256:validatorSha})+'\n',{mode:0o600});
 const attemptStore=join(dir,'attempt-store');await mkdir(join(attemptStore,'attempts'),{recursive:true});
 const cfg={binary:join(root,'.voder-build/release/voder-gateway'),daemonBinary:join(root,'.voder-build/release/voderd'),policyFile,attemptStore,nodeBinary:node,modalCli:modal,targetPath:'src/input.txt',timeoutMs:360000};
 await writeFile(join(dir,'host-config.json'),JSON.stringify(cfg)+'\n',{mode:0o600});
 const result={schema:'voder.sdk.real.native.postcommit.crash.v1',dir,real_model_called:false,real_modal_may_be_billed:true,source_modified:false,pass:false};
 let child;
 try{
  child=spawn(node,[new URL(import.meta.url).pathname,'child',dir],{detached:false,stdio:['ignore','ignore','pipe'],shell:false});
  const childClosed=new Promise(resolve=>child.once('close',(code,signal)=>resolve({code,signal})));
  let errorLog='';child.stderr.on('data',c=>{errorLog=(errorLog+c.toString('utf8')).slice(-4000)});
  let markerPayload=null;const deadline=Date.now()+450000;
  while(Date.now()<deadline){
   try{markerPayload=JSON.parse(await readFile(join(dir,'native-committed.marker'),'utf8'));break;}
   catch{if(child.exitCode!==null||child.signalCode!==null)throw Error('SDK_CHILD_EXITED_BEFORE_NATIVE_COMMIT '+errorLog.slice(-300));await new Promise(r=>setTimeout(r,200));}
  }
  if(!markerPayload)throw Error('SDK_NATIVE_GATEWAY_COMMIT_TIMEOUT');
  const close=await childClosed;
  if(close.signal!=='SIGKILL')throw Error('SDK_NOT_KILLED_AFTER_NATIVE_COMMIT');
  const sessionId=(await readFile(join(dir,'session-id'),'utf8')).trim();
  if(markerPayload.session_id!==sessionId)throw Error('SESSION_IDENTITY_MISMATCH');
  const journal=await new VoderJournal({dir:join(dir,'journal'),workspace:repo}).initialize();
  const crashed=await journal.get(sessionId);
  if(crashed.status!=='INSPECTING'||crashed.native_actions?.length!==1||crashed.native_actions[0].status!=='DISPATCH_INTENT'||crashed.native_actions[0].action_id!==markerPayload.action_id)
   throw Error('DURABLE_ACTION_INTENT_MISSING');
  let newModelCalls=0;
  const modelHost={async createSession(){newModelCalls++;throw Error('REPLAY_MODEL_FORBIDDEN');}};
  const sdk=new VoderSDK({cwd:repo,nativeGateway:cfg,modelHost});
  const nativeContinuation=new NativeSdkContinuation({gatewayBinary:cfg.binary,daemonBinary:cfg.daemonBinary,attemptStore:cfg.attemptStore});
  const ctrl=new VoderControlPlane({sdk,journal,nativeContinuation});
  const reconciled=await ctrl.autoRecover({taskResolver:async()=>task});
  if(reconciled.length!==1||reconciled[0].status!=='RECOVERY_REVIEW_REQUIRED'||reconciled[0].reason!=='NATIVE_ACTION_COMMITTED_NO_AUTOREPLAY'||newModelCalls!==0)
   throw Error('DID_NOT_FAIL_CLOSED_ON_REAL_RUST_COMMIT '+JSON.stringify(reconciled));
  if(git(repo,'status','--porcelain')!=='')throw Error('ORIGINAL_REPOSITORY_MUTATED');
  result.pass=true;result.rust_postcommit_verification='PASS';result.auto_replay='PREVENTED';
  console.log('SDK_POSTCOMMIT_SIGKILL_RUST_E2E=PASS no replay, independent Rust commit verified');
  ctrl.dispose();
 }catch(e){result.failure={code:e.code??'ERROR',message:String(e.message??e).slice(0,550)};console.error('SDK_POSTCOMMIT_SIGKILL_RUST_E2E=NOT_PASSED '+result.failure.message);}
 finally{if(child && child.exitCode===null && child.signalCode===null)child.kill('SIGKILL');
  await writeFile(join(dir,'ACCEPTANCE.json'),JSON.stringify(result,null,2)+'\n',{mode:0o600});console.log('SDK_REAL_NATIVE_EVIDENCE='+dir);}
 if(!result.pass)process.exitCode=4;
}
if(process.argv[2]==='child')await runChild();else await parent();
