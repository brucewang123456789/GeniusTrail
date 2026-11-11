/** Recoverable all-or-none Voder delivery transaction; arbitrary external filesystem readers do not share Voder locks. */
import {readFile,mkdir,lstat,rename,rmdir,unlink} from 'node:fs/promises';import {resolve,join} from 'node:path';import {createHash,randomUUID} from 'node:crypto';
import {normalizedCandidate} from '../../voder-orchestration/src/index.mjs';import {VoderError} from '../../voder-agent-core/src/contracts.mjs';
import {noLinks,regularBytes,writeNew,durableJson,fsyncDirectory,filePath} from './verified-delivery.mjs';
const sha=x=>createHash('sha256').update(x).digest('hex'),fail=(c,m)=>{throw new VoderError(c,m);};
const id=/^[A-Za-z0-9._-]{1,64}$/,hex=/^[a-f0-9]{64}$/;
async function load(path){await noLinks(path);const b=await readFile(path);if(b.length>32768)fail('BATCH_PROOF_SIZE','Oversized transaction evidence');return JSON.parse(b);}
async function replaceExact(path,expected,data,mode){const now=await regularBytes(path);if(sha(now.data)!==expected)fail('DELIVERY_CONFLICT','Transaction member diverged');const temp=path+'.voder-'+randomUUID();
 try{await writeNew(temp,data,mode);const again=await regularBytes(path);if(again.ino!==now.ino||again.dev!==now.dev||!again.data.equals(now.data))fail('DELIVERY_CONFLICT','Transaction member raced');await rename(temp,path);await fsyncDirectory(resolve(path,'..'));}finally{await unlink(temp).catch(()=>{});}}
async function owner(){const boot=(await readFile('/proc/sys/kernel/random/boot_id','utf8')).trim(),stat=await readFile('/proc/self/stat','utf8');return {pid:process.pid,boot,ticks:stat.slice(stat.lastIndexOf(')')+2).split(/\s+/)[19]};}
async function alive(o){if(!o||!Number.isSafeInteger(o.pid)||typeof o.boot!=='string'||typeof o.ticks!=='string')fail('BATCH_RECOVERY_IDENTITY','Invalid original writer identity');try{const boot=(await readFile('/proc/sys/kernel/random/boot_id','utf8')).trim(),stat=await readFile('/proc/'+o.pid+'/stat','utf8');return boot===o.boot&&stat.slice(stat.lastIndexOf(')')+2).split(/\s+/)[19]===o.ticks;}catch(e){if(e.code==='ENOENT')return false;throw e;}}
export class BatchWorkspaceDelivery {
 constructor({cwd,attemptStore,policyFile}={}){if(![cwd,attemptStore,policyFile].every(x=>typeof x==='string'&&x.startsWith('/')))fail('DELIVERY_CONFIG','Absolute host paths required');this.cwd=resolve(cwd);this.attemptStore=resolve(attemptStore);this.policyFile=resolve(policyFile);}
 async _checkpoint(){}
 async _policy(){const p=await load(this.policyFile);if(p.schema!=='voder.gateway.policy.v2'||resolve(p.repository_path)!==this.cwd||!Array.isArray(p.target_paths)||p.target_paths.length<2||p.target_paths.length>16||p.target_paths.some((x,i)=>typeof x!=='string'||(i&&p.target_paths[i-1]>=x)))fail('DELIVERY_POLICY','Exact operator batch policy required');for(const path of p.target_paths)filePath(this.cwd,path);return p;}
 async _locks(paths){await noLinks(this.attemptStore);const dir=join(this.attemptStore,'verified-delivery-locks');try{await mkdir(dir,{mode:0o700});}catch(e){if(e.code!=='EEXIST')throw e;}await noLinks(dir);const meta=await lstat(dir);if(!meta.isDirectory()||(meta.mode&0o077))fail('DELIVERY_LOCK_UNSAFE','Private lock root required');const held=[];
  try{for(const path of paths){const lock=join(dir,sha(this.cwd+'\0'+path));try{await mkdir(lock,{mode:0o700});}catch(e){if(e.code==='EEXIST')fail('DELIVERY_LOCKED','Batch member locked');throw e;}held.push(lock);}return held;}
  catch(e){for(const lock of held.reverse())await rmdir(lock);throw e;}
 }
 async _restore(proofDir,proof){
  const states=[];
  for(let i=0;i<proof.files.length;i++){const row=proof.files[i],path=filePath(this.cwd,row.relative_path),current=await regularBytes(path),before=await regularBytes(join(proofDir,'before-'+i+'.bin'));
   if(sha(before.data)!==row.before_sha256||![row.before_sha256,row.after_sha256].includes(sha(current.data)))fail('BATCH_RECOVERY_REVIEW_REQUIRED','Unknown member bytes; do not overwrite user changes');states.push({row,path,before,current});}
  for(const {row,path,before,current} of states.reverse())if(sha(current.data)!==row.before_sha256)await replaceExact(path,row.after_sha256,before.data,before.mode);
  for(const row of proof.files)if(sha((await regularBytes(filePath(this.cwd,row.relative_path))).data)!==row.before_sha256)fail('BATCH_RECOVERY_REVIEW_REQUIRED','Restoration postcheck failed');
  await durableJson(join(proofDir,'rolled-back.json'),{...proof,status:'ROLLED_BACK_DELIVERY'});
 }
 async deliver({input,receipt,signal}={}){
  const candidate=normalizedCandidate({patches:input?.patches});const policy=await this._policy();
  if(input?.kind!=='patch_batch'||!id.test(input.action_id??'')||['.','..'].includes(input.action_id)||receipt?.authority!=='VODER_NATIVE_VGR'||receipt.status!=='COMMITTED'||receipt.session_id!==input.session_id||receipt.action_id!==input.action_id||receipt.run_id!==`run-${input.action_id}`||!hex.test(receipt.commit_digest??'')||!hex.test(receipt.journal_digest??'')||JSON.stringify(candidate.patches.map(p=>p.relative_path))!==JSON.stringify(policy.target_paths))fail('DELIVERY_UNVERIFIED','Exact batch Rust commit and policy required');
  const actionRoot=join(this.attemptStore,'attempts',input.action_id),proofDir=join(actionRoot,'verified-delivery'),candidateRoot=join(actionRoot,'owned-worktrees','candidate');await noLinks(candidateRoot);
  if(resolve(receipt.workspace_path??'/')!==candidateRoot)fail('DELIVERY_CANDIDATE','Native candidate root mismatch');
  const held=await this._locks(policy.target_paths);let begun=false,complete=false,restored=false,proof;
  try{
   await noLinks(this.cwd);await noLinks(actionRoot);
   // Existing intent is a no-replay fence, never a request to continue an interrupted write.
   await mkdir(proofDir,{mode:0o700});begun=true;await fsyncDirectory(actionRoot);
   const rows=[];
   for(let i=0;i<candidate.patches.length;i++){const p=candidate.patches[i],before=await regularBytes(filePath(this.cwd,p.relative_path)),verified=await regularBytes(filePath(candidateRoot,p.relative_path));
    if(sha(before.data)!==p.old_sha256||!verified.data.equals(Buffer.from(p.new_content)))fail('DELIVERY_CONFLICT','Batch target or verified candidate differs');
    await writeNew(join(proofDir,'before-'+i+'.bin'),before.data,before.mode);rows.push({relative_path:p.relative_path,before_sha256:p.old_sha256,after_sha256:sha(verified.data)});
   }
   proof={schema:'voder.verified-batch-delivery.v1',status:'INTENT',owner:await owner(),session_id:input.session_id,action_id:input.action_id,run_id:receipt.run_id,commit_digest:receipt.commit_digest,journal_digest:receipt.journal_digest,files:rows};
   await durableJson(join(proofDir,'intent.json'),proof);
   for(const [index,p] of candidate.patches.entries()){if(signal?.aborted)fail('DELIVERY_CANCELLED','Batch cancelled before complete delivery');await replaceExact(filePath(this.cwd,p.relative_path),p.old_sha256,Buffer.from(p.new_content),(await regularBytes(filePath(this.cwd,p.relative_path))).mode);await this._checkpoint('MEMBER_WRITTEN',index);}
   for(const r of rows)if(sha((await regularBytes(filePath(this.cwd,r.relative_path))).data)!==r.after_sha256)fail('DELIVERY_POSTCHECK','Batch postcheck failed');
   await durableJson(join(proofDir,'delivered.json'),{...proof,status:'DELIVERED'});complete=true;return {status:'DELIVERED',action_id:input.action_id,commit_digest:receipt.commit_digest,files:rows};
  }catch(e){if(proof){try{await this._restore(proofDir,proof);restored=true;}catch{fail('BATCH_RECOVERY_REVIEW_REQUIRED','Interrupted batch remains fenced; operator recovery required');}}throw e;}
  finally{if(complete||restored||!proof)for(const lock of held.reverse())await rmdir(lock);}
 }
 async reconcile({actionId,sessionId,commitVerifier}={}){
  if(!id.test(actionId??'')||['.','..'].includes(actionId)||!commitVerifier?.confirmCommit)fail('BATCH_RECOVERY_IDENTITY','Operator identity and independent verifier required');
  const policy=await this._policy(),dir=join(this.attemptStore,'attempts',actionId,'verified-delivery'),proof=await load(join(dir,'intent.json'));
  if((sessionId!==undefined&&proof.session_id!==sessionId)||proof.schema!=='voder.verified-batch-delivery.v1'||proof.action_id!==actionId||proof.run_id!==`run-${actionId}`||!Array.isArray(proof.files)||JSON.stringify(proof.files.map(x=>x.relative_path))!==JSON.stringify(policy.target_paths)||proof.files.some(x=>!hex.test(x.before_sha256??'')||!hex.test(x.after_sha256??'')))fail('BATCH_RECOVERY_IDENTITY','Invalid batch intent');
  if(await alive(proof.owner))fail('BATCH_OWNER_STILL_LIVE','Original delivery process still running');
  const recoveryLock=join(dir,'recovery.lock');await mkdir(recoveryLock,{mode:0o700});
  try{
  const observed=await commitVerifier.confirmCommit({...proof,authority:'VODER_NATIVE_VGR',status:'COMMITTED'},{session_id:proof.session_id,action_id:actionId});
  if(observed?.verified!==true)fail('BATCH_RECOVERY_UNVERIFIED','Rust commit not confirmed');
  try{await load(join(dir,'delivered.json'));return {status:'ALREADY_DELIVERED',delivery_allowed:false,native_replay:false};}catch(e){if(e.code!=='ENOENT')throw e;}
  try{await load(join(dir,'rolled-back.json'));return {status:'ALREADY_RESTORED',delivery_allowed:false,native_replay:false};}catch(e){if(e.code!=='ENOENT')throw e;}
  await this._restore(dir,proof);
  for(const path of policy.target_paths)await rmdir(join(this.attemptStore,'verified-delivery-locks',sha(this.cwd+'\0'+path)));
  return {status:'ROLLED_BACK_DELIVERY',native_commit_preserved:true,delivery_allowed:false,native_replay:false};
  }finally{await rmdir(recoveryLock);}
 }
}
export async function readBatchDiff({dir,workspace,id,action,run,commit,policyFile,commitVerifier}){
 const proof=await load(join(dir,'delivered.json')),intent=await load(join(dir,'intent.json')),policy=await load(policyFile);
 if(proof.schema!=='voder.verified-batch-delivery.v1'||proof.status!=='DELIVERED'||proof.session_id!==id||proof.action_id!==action||proof.run_id!==run||proof.commit_digest!==commit||JSON.stringify({...proof,status:'INTENT'})!==JSON.stringify(intent)||policy.schema!=='voder.gateway.policy.v2'||resolve(policy.repository_path)!==workspace||JSON.stringify(proof.files?.map(x=>x.relative_path))!==JSON.stringify(policy.target_paths))fail('DIFF_PROOF_MISMATCH','Batch proof differs from exact native session');
 const authoritative=await commitVerifier.confirmCommit({...proof,status:'COMMITTED',authority:'VODER_NATIVE_VGR'},{session_id:id,action_id:action});if(authoritative.verified!==true)fail('DIFF_RUST_NOT_VERIFIED','Batch commit not independently confirmed');
 const files=[];let total=0;
 for(let i=0;i<proof.files.length;i++){const row=proof.files[i],before=(await regularBytes(join(dir,'before-'+i+'.bin'))).data,after=(await regularBytes(filePath(workspace,row.relative_path))).data,candidate=(await regularBytes(filePath(join(resolve(dir,'..'),'owned-worktrees','candidate'),row.relative_path))).data;
  total+=before.length+after.length;if(total>255000||sha(before)!==row.before_sha256||sha(after)!==row.after_sha256||!after.equals(candidate))fail('DIFF_WORKTREE_CHANGED','Batch bytes changed or review budget exceeded');
  const decoder=new TextDecoder('utf8',{fatal:true});files.push({...row,before:decoder.decode(before),after:decoder.decode(after)});
 }
 return {schema:'voder.session.diff.v2',status:'VERIFIED_DELIVERED',sessionId:id,authority:'VODER_NATIVE_VGR_STATE',action_id:action,run_id:run,commit_digest:commit,files};
}
