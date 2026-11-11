#!/usr/bin/env node
/** Cloud-only Rust Journal observation of prior real SIGKILL rollbacks and final commits.
    Stages copies in repair. No gateway dispatch, no model or Modal execution. */
import {mkdtemp,cp,mkdir,readdir,readFile,writeFile,lstat,stat,chmod} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {randomUUID,createHash} from 'node:crypto';
import {NativeSdkContinuation} from '../packages/voder-sdk/src/native-continuation.mjs';
const root=resolve(process.env.VODER_ROOT||'/root/autodl-tmp');
const repair=join(root,'repair');
const exact=process.env.VODER_T08_EVIDENCE;
const T08=exact?resolve(exact):join(repair,'.voder-t08-node-realpath-jn5LFGnJ');
const committedRoot=resolve(process.env.VODER_NATIVE_COMMITTED_EVIDENCE||join(repair,'.voder-full-e2e-O7IIlT'));
const reportDir=await mkdtemp(join(repair,'.voder-sdk-rust-observation-'));
await chmod(reportDir,0o700);
const daemon=join(root,'.voder-build/release/voderd'),gateway=join(root,'.voder-build/release/voder-gateway');
const report={schema:'voder.sdk.native.observation.v1',checked:[],authority:'REAL_RUST_READ_ONLY',source_modified:false,modal_called:false,success:false};
const safeDir=async p=>{const st=await lstat(p);if(!st.isDirectory()||st.isSymbolicLink())throw Error('UNSAFE_DIR: '+p);return p};
async function stage(original,label,ident){
 await safeDir(original);
 const place=join(reportDir,label),store=join(place,'store');
 const target=join(store,'attempts',ident.action_id,'daemon-state');
 await mkdir(target,{recursive:true,mode:0o700});
 await cp(original,target,{recursive:true,force:false,errorOnExist:false,filter:async(source)=>{
  const st=await lstat(source);if(st.isSymbolicLink())throw Error('UNSAFE_SYMLINK_IN_RUST_EVIDENCE');return true;
 }});
 const recovery=new NativeSdkContinuation({gatewayBinary:gateway,daemonBinary:daemon,attemptStore:store});
 // Explicitly READ-ONLY: bypass privileged reconcile() which can call --recover.
 const verdict=await recovery._observe(ident);
 report.checked.push({label,action_id:ident.action_id,status:verdict.status,authority:verdict.authority,original});
 return verdict;
}
try{
 const pre=JSON.parse(await readFile(join(T08,'CRASH.json'),'utf8'));
 if(pre.pass!==true||!Array.isArray(pre.cases)||pre.cases.length<2)throw Error('HISTORICAL_T08_NOT_PASSED');
 for(const phase of ['POST_PATCH','POST_META_AUDIT']){
  const c=pre.cases.find(x=>x.phase===phase&&x.status==='PASS'&&x.recovery==='RUST_MEASURED_ROLLBACK');
  if(!c)throw Error('T08_REQUIRED_PHASE_NOT_PASSED_'+phase);
  const parent=join(T08,'crash-'+phase.toLowerCase());await safeDir(parent);
  const roots=(await readdir(parent)).filter(x=>x.startsWith('voder-vgr6-real-modal-'));
  if(roots.length!==1)throw Error('AMBIGUOUS_REAL_RUST_CHECKPOINT');
  const stateDir=join(parent,roots[0],'daemon-state');
  const status=await stage(stateDir,phase.toLowerCase(),{session_id:'vgr6-accept-session',run_id:'vgr6-accept-run',action_id:'vgr6-accept-action'});
  if(status.status!=='ROLLED_BACK'||status.authority!=='VODER_NATIVE_VGR_RECOVERY'||status.delivery_allowed!==false)
   throw Error('RUST_ROLLBACK_NOT_VERIFIED_'+phase);
  console.log(`SDK_REAL_RUST_OBSERVER_${phase}=PASS`);
 }
 // Observe one distinct REAL committed Rust run, with no new action submission.
 const dirs=await readdir(committedRoot);
 const caseNames=dirs.filter(x=>x.endsWith('-model-cli')||x.endsWith('-native-happy')).sort();
 if(!caseNames.length)throw Error('NATIVE_COMMIT_EVIDENCE_NOT_FOUND');
 let found=false;
 for(const name of caseNames){
  const attempts=join(committedRoot,name,'attempt-store','attempts');
  let names;try{names=await readdir(attempts);}catch{continue;}
  for(const action of names.filter(x=>/^action-/.test(x))){
   const stateDir=join(attempts,action,'daemon-state');
   let events;try{events=(await readFile(join(stateDir,'events.jsonl'),'utf8')).split('\n').filter(Boolean).map(JSON.parse);}catch{continue;}
   const final=events.map(x=>x.body??x).find(x=>x.type==='vgr.final_committed'&&x.action_id===action);
   if(!final)continue;
   const status=await stage(stateDir,'committed', {session_id:final.session,run_id:final.run_id,action_id:action});
   if(status.status!=='COMMITTED'||status.authority!=='VODER_NATIVE_VGR_STATE')throw Error('COMMITTED_RUST_STATUS_NOT_VERIFIED');
   found=true;break;
  }
  if(found)break;
 }
 if(!found)throw Error('HISTORICAL_COMMITTED_VGR_NOT_VERIFIED');
 console.log('SDK_REAL_RUST_OBSERVER_COMMITTED=PASS');
 report.success=true;
}catch(e){report.failure={code:e?.code??'EVIDENCE_BLOCKED',detail:String(e?.message??e).slice(0,500)};console.error('SDK_REAL_RUST_OBSERVATION=NOT_PASSED '+report.failure.detail);}
const reportFile=join(reportDir,'ACCEPTANCE.json');
await writeFile(reportFile,JSON.stringify(report,null,2)+'\n',{mode:0o600});
console.log('SDK_NATIVE_OBSERVATION_REPORT='+reportFile);
console.log('SDK_REAL_RUST_OBSERVATION='+ (report.success?'PASS':'NOT_PASSED'));
process.exitCode=report.success?0:4;
