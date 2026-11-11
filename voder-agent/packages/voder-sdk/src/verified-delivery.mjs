/** Host-owned, single-file, crash-auditable delivery following an independently verified Rust VGR commit.
 * This never authorizes execution, replays an action, or accepts model-selected output paths.
 */
import {createHash,randomUUID} from 'node:crypto';
import {constants} from 'node:fs';
import {open,readFile,realpath,lstat,mkdir,rename,unlink,rmdir} from 'node:fs/promises';
import {resolve,join,dirname,relative,sep,parse} from 'node:path';
import {VoderError,isAuthorizedSourcePath} from '../../voder-agent-core/src/contracts.mjs';

const SHA=/^[0-9a-f]{64}$/;
const ID=/^[A-Za-z0-9._-]{1,64}$/;
const sha=data=>createHash('sha256').update(data).digest('hex');
const reject=(code,message)=>{throw new VoderError(code,message)};
const filePath=(root,rel)=>{
 if(!isAuthorizedSourcePath(rel))
   reject('DELIVERY_SCOPE','Unsafe target path');
 const p=resolve(root,rel);if(!p.startsWith(resolve(root)+sep))reject('DELIVERY_SCOPE','Target escaped workspace');return p;
};
async function noLinks(path){
 const abs=resolve(path),top=parse(abs).root;let cursor=top;
 for(const part of relative(top,abs).split(sep).filter(Boolean)){
  cursor=join(cursor,part);const s=await lstat(cursor);
  if(s.isSymbolicLink())reject('DELIVERY_SYMLINK','Symlink in delivery path');
 }
}
async function regularBytes(path){
 await noLinks(path);
 const h=await open(path,constants.O_RDONLY|constants.O_NOFOLLOW);
 try{
  const st=await h.stat();if(!st.isFile())reject('DELIVERY_FILE_TYPE','Expected regular file');
  return {data:await h.readFile(),mode:st.mode&0o777,dev:st.dev,ino:st.ino};
 }finally{await h.close()}
}
async function fsyncDirectory(dir){const h=await open(dir,'r');try{await h.sync()}finally{await h.close()}}
async function writeNew(path,bytes,mode=0o600){
 const h=await open(path,constants.O_WRONLY|constants.O_CREAT|constants.O_EXCL|constants.O_NOFOLLOW,mode);
 try{await h.writeFile(bytes);await h.chmod(mode);await h.sync()}finally{await h.close()}
}
async function durableJson(path,obj){await writeNew(path,Buffer.from(JSON.stringify(obj)+'\n'));await fsyncDirectory(dirname(path))}
/** Once the write INTENT exists, a crash requires explicit reconciliation; there is no automatic replay. */
export class VerifiedWorkspaceDelivery {
 constructor({cwd,attemptStore,policyFile}={}){
  if(!cwd||!attemptStore||!policyFile||![cwd,attemptStore,policyFile].every(p=>typeof p==='string'&&p.startsWith('/')))
    reject('DELIVERY_CONFIG','Only explicit absolute host-owned paths accepted');
  this.cwd=resolve(cwd);this.attemptStore=resolve(attemptStore);this.policyFile=resolve(policyFile);
 }
 async deliver({input,receipt,signal}={}){
  if(!input||!receipt||receipt.status!=='COMMITTED'||receipt.authority!=='VODER_NATIVE_VGR'||
     receipt.session_id!==input.session_id||receipt.action_id!==input.action_id||receipt.run_id!==`run-${input.action_id}`||
     (!ID.test(input.action_id??'')||['.','..'].includes(input.action_id))||!SHA.test(receipt.commit_digest??'')||!SHA.test(receipt.journal_digest??'')||
     input.kind!=='patch'||!SHA.test(input.old_sha256??'')||typeof input.new_content!=='string')
   reject('DELIVERY_UNVERIFIED','Exact independently confirmed Rust commit required');
  // Lock the exact target across independent action attempts. A crash retains the
  // lock for manual reconciliation; it never silently enables a second write.
  await noLinks(this.attemptStore);
  const locks=join(this.attemptStore,'verified-delivery-locks');
  try{await mkdir(locks,{mode:0o700})}catch(e){if(e.code!=='EEXIST')throw e}
  await noLinks(locks);
  const locksStat=await lstat(locks);
  if(!locksStat.isDirectory()||(locksStat.mode&0o077))reject('DELIVERY_LOCK_UNSAFE','Private lock directory required');
  const lock=join(locks,sha(Buffer.from(this.cwd+'\0'+input.relative_path)));
  try{await mkdir(lock,{mode:0o700})}catch(e){if(e.code==='EEXIST')reject('DELIVERY_LOCKED','Target delivery lock is held or requires crash reconciliation');throw e}
  let delivered=false;
  try{const result=await this._deliverOnce({input,receipt,signal});delivered=true;return result}
  finally{
   // An interrupted INTENT is an ambiguous operation: retain its target lock for
   // operator reconciliation; do not permit another action to overwrite it.
   const proof=join(this.attemptStore,'attempts',input.action_id,'verified-delivery');
   const began=await lstat(proof).then(()=>true,e=>{if(e.code==='ENOENT')return false;return true});
   if(delivered||!began)await rmdir(lock).catch(()=>{});
  }
 }
 async _deliverOnce({input,receipt,signal}){
  // Caller is the trusted SDK broker, invoked exclusively AFTER commitVerifier.confirmCommit.
  if(signal?.aborted)reject('DELIVERY_CANCELLED','Cancelled before workspace delivery');
  await noLinks(this.policyFile);const policy=JSON.parse(await readFile(this.policyFile,'utf8'));
  if(policy?.schema!=='voder.gateway.policy.v1'||policy.target_path!==input.relative_path||
     typeof policy.repository_path!=='string'||resolve(policy.repository_path)!==this.cwd)
   reject('DELIVERY_POLICY','Rust gateway policy does not name this exact source root and file');
  await noLinks(this.cwd);
  if(await realpath(this.cwd)!==this.cwd)reject('DELIVERY_SCOPE','Workspace must use canonical, non-symlink path');
  const actionRoot=join(this.attemptStore,'attempts',input.action_id);
  const candidateRoot=join(actionRoot,'owned-worktrees','candidate');
  await noLinks(candidateRoot);
  if(!receipt.workspace_path||await realpath(candidateRoot)!==await realpath(receipt.workspace_path))
   reject('DELIVERY_CANDIDATE','Trusted candidate worktree differs from native commit receipt');
  const proofDir=join(actionRoot,'verified-delivery');
  if(await lstat(proofDir).then(()=>true,e=>{if(e.code==='ENOENT')return false;throw e}))
    reject('DELIVERY_ALREADY_ATTEMPTED','Never replay an existing delivery transaction');
  const target=filePath(this.cwd,input.relative_path),candidate=filePath(candidateRoot,input.relative_path);
  const proposed=Buffer.from(input.new_content,'utf8');
  if(!proposed.length)reject('DELIVERY_BOUNDS','Invalid replacement size');
  const targetBefore=await regularBytes(target),candidateState=await regularBytes(candidate);
  if(sha(targetBefore.data)!==input.old_sha256)reject('DELIVERY_CONFLICT','User file no longer matches model preimage');
  if(!candidateState.data.equals(proposed))reject('DELIVERY_CANDIDATE','Candidate differs from verified proposal bytes');
  // Dedicated no-clobber on-disk intent doubles as a one-shot delivery fence across processes.
  await noLinks(actionRoot);
  try{await mkdir(proofDir,{mode:0o700})}catch(e){if(e?.code==='EEXIST')reject('DELIVERY_ALREADY_ATTEMPTED','Never replay an existing delivery transaction');throw e}
  await fsyncDirectory(actionRoot);
  const proof={schema:'voder.verified-delivery.v1',session_id:input.session_id,action_id:input.action_id,
    run_id:receipt.run_id,commit_digest:receipt.commit_digest,journal_digest:receipt.journal_digest,
    relative_path:input.relative_path,before_sha256:input.old_sha256,after_sha256:sha(proposed),status:'INTENT'};
  await durableJson(join(proofDir,'intent.json'),proof);
  // Durable before-image permits operator recovery without replaying the native action.
  await writeNew(join(proofDir,'before.bin'),targetBefore.data);
  await fsyncDirectory(proofDir);
  // Atomic single-file replacement; verify twice before rename and refuse unexpected inode/content changes.
  const staged=join(dirname(target),`.voder-delivery-${randomUUID()}`);
  let renamed=false;
  try{
   await writeNew(staged,proposed,targetBefore.mode);
   const again=await regularBytes(target);
   if(sha(again.data)!==proof.before_sha256||again.ino!==targetBefore.ino||again.dev!==targetBefore.dev)
     reject('DELIVERY_CONFLICT','User file changed during delivery preparation');
   if(signal?.aborted)reject('DELIVERY_CANCELLED','Cancelled before workspace mutation');
   await rename(staged,target);renamed=true;await fsyncDirectory(dirname(target));
   const after=await regularBytes(target);
   if(sha(after.data)!==proof.after_sha256)reject('DELIVERY_POSTCHECK','Post-delivery bytes mismatch');
   await durableJson(join(proofDir,'delivered.json'),{...proof,status:'DELIVERED'});
   return Object.freeze({status:'DELIVERED',action_id:input.action_id,commit_digest:receipt.commit_digest,
     before_sha256:proof.before_sha256,after_sha256:proof.after_sha256});
  }finally{if(!renamed)await unlink(staged).catch(()=>{})}
 }
}

export {noLinks,regularBytes,writeNew,durableJson,fsyncDirectory,filePath};
