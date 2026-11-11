'use strict';
/** Host-owned final publication: exact VGR-delivered artifacts, second explicit user approval,
 *  conflict detection against the private-copy baseline, and no implicit retries.
 *  It never invokes a model, sandbox, native action, repository hook or benchmark workflow.
 */
const fs=require('node:fs/promises');
const {constants:C}=require('node:fs');
const path=require('node:path');
const crypto=require('node:crypto');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const fail=(code,message)=>{throw Object.assign(Error(message),{code});};
const SHA=/^[a-f0-9]{64}$/;
const UUID=/^[0-9a-f]{8}-[0-9a-f-]{27,}$/i;
const RESERVED=new Set(['.git','.voder-tools','.voder-build','node_modules']);
function relativeFile(value){
 if(typeof value!=='string'||value.length>240||value.includes('\\')||value.startsWith('/')||value.includes('\0'))fail('PUBLICATION_PATH','Invalid source-relative path');
 const parts=value.split('/');
 if(!parts.length||parts.some(p=>!p||p==='.'||p==='..'||RESERVED.has(p)||!/^[A-Za-z0-9_.-]+$/.test(p)))fail('PUBLICATION_PATH','Unsafe source-relative path');
 return value;
}
// A baseline inventories unrelated user assets too (spaces and Unicode are
// valid filenames). This never expands the stricter artifact publication scope.
function baselineFile(value){
 if(typeof value!=='string'||!value||value.includes('\\')||value.startsWith('/')||/[\u0000-\u001f\u007f]/.test(value))fail('PUBLICATION_MANIFEST','Invalid baseline path');
 if(value.split('/').some(p=>!p||p==='.'||p==='..'||RESERVED.has(p)))fail('PUBLICATION_MANIFEST','Unsafe baseline path');
 return value;
}
async function pathState(p,{directory=false,allowAbsent=false,owner=false}={}){
 const absolute=path.resolve(p),parts=absolute.split(path.sep).filter(Boolean);
 let cursor=path.parse(absolute).root;
 for(let i=0;i<parts.length;i++){
  cursor=path.join(cursor,parts[i]);let s;
  try{s=await fs.lstat(cursor);}catch(e){if(e.code==='ENOENT'&&allowAbsent)return null;throw e;}
  if(s.isSymbolicLink())fail('PUBLICATION_LINK','Symlink in publication path');
  if(i<parts.length-1||directory){if(!s.isDirectory())fail('PUBLICATION_NOT_DIRECTORY','Expected directory in publication path');}
  if((s.mode&0o022)!==0)fail('PUBLICATION_UNSAFE_PERMISSIONS','Shared writable path is not an authorized destination');
  if(owner&&i===parts.length-1&&process.getuid&&s.uid!==process.getuid())fail('PUBLICATION_OWNER','Destination must be owned by current user');
  if(i===parts.length-1)return s;
 }
 return null;
}
async function bytes(p,limit=8*1024*1024){
 const s=await pathState(p);if(!s||!s.isFile()||s.nlink!==1||s.size>limit)fail('PUBLICATION_FILE_TYPE','Unsupported publication file');
 const h=await fs.open(p,C.O_RDONLY|C.O_NOFOLLOW);
 try{const before=await h.stat();if(before.ino!==s.ino||before.dev!==s.dev)fail('PUBLICATION_RACE','File changed while opening');
 const b=await h.readFile();const after=await h.stat();if(after.ino!==before.ino||after.dev!==before.dev||after.size!==b.length||after.mtimeMs!==before.mtimeMs)fail('PUBLICATION_RACE','File changed while reading');
 return {data:b,stat:s};}finally{await h.close();}
}
async function syncDir(p){const h=await fs.open(p,'r');try{await h.sync();}finally{await h.close();}}
async function newFile(p,data,mode=0o600){const h=await fs.open(p,C.O_CREAT|C.O_EXCL|C.O_WRONLY|C.O_NOFOLLOW,mode);try{await h.writeFile(data);await h.sync();}finally{await h.close();}}
// Shared directories are reusable; a file or symlink is never a directory.
async function ensurePublicationDirectory(p){
 await pathState(p,{directory:true,allowAbsent:true,owner:true});
 try{await fs.mkdir(p,{mode:0o700});}catch(e){if(e.code!=='EEXIST')throw e;}
 await pathState(p,{directory:true,owner:true});
}
async function manifestFor({originalRoot,privateRoot,storageRoot}){
 const canonical=path.resolve(originalRoot),isolated=path.resolve(privateRoot),storage=path.resolve(storageRoot);
 if(canonical===isolated||!isolated.startsWith(storage+path.sep)||path.dirname(isolated)!==path.join(storage,'working-copies')||!UUID.test(path.basename(isolated)))fail('PUBLICATION_IDENTITY','Only a private copy created by Voder can be published');
 await pathState(canonical,{directory:true,owner:true});await pathState(isolated,{directory:true,owner:true});
 const manifestPath=path.join(storage,'working-copies',path.basename(isolated)+'.json');
 const m=await bytes(manifestPath,2*1024*1024);
 if((m.stat.mode&0o077)!==0)fail('PUBLICATION_MANIFEST','Private baseline record required');
 let obj;try{obj=JSON.parse(m.data);}catch{fail('PUBLICATION_MANIFEST','Invalid baseline JSON');}
 if(obj.schema!=='voder.workspace-copy.v1'||obj.source!==canonical||obj.root!==isolated||!Array.isArray(obj.files)||obj.files.length>4096)fail('PUBLICATION_MANIFEST','Baseline identity differs');
 const baseline=new Map();for(const row of obj.files){baselineFile(row.path);if(!SHA.test(row.sha256)||baseline.has(row.path))fail('PUBLICATION_MANIFEST','Invalid baseline entry');baseline.set(row.path,row.sha256);}
 return baseline;
}
async function publishOriginalProject({originalRoot,privateRoot,storageRoot,sessionId,result,approve}={}){
 if(typeof approve!=='function'||typeof sessionId!=='string'||!UUID.test(sessionId))fail('PUBLICATION_AUTH','Trusted task approval and session identity required');
 if(result?.status!=='DELIVERED'||result?.verified!==true||result?.taskAcceptance?.schema!=='voder.task-acceptance.v1'||result.taskAcceptance.delivery!=='DELIVERED')fail('PUBLICATION_UNVERIFIED','Native verified private delivery is required');
 const original=path.resolve(originalRoot),privatePath=path.resolve(privateRoot);
 const baseline=await manifestFor({originalRoot:original,privateRoot:privatePath,storageRoot});
 const artifacts=result.taskAcceptance.artifacts;
 if(!Array.isArray(artifacts)||artifacts.length<1||artifacts.length>16)fail('PUBLICATION_SCOPE','Expected 1–16 independently recorded artifacts');
 const prepared=[],seen=new Set();
 for(const item of artifacts){
  const rel=relativeFile(item.path);if(seen.has(rel)||!SHA.test(item.sha256))fail('PUBLICATION_SCOPE','Duplicate or unverified artifact');seen.add(rel);
  const privateFile=path.join(privatePath,rel);const b=await bytes(privateFile);
  if(sha(b.data)!==item.sha256)fail('PUBLICATION_DIGEST','Artifact changed after Rust delivery');
  const dest=path.join(original,rel),target=await pathState(dest,{allowAbsent:true});
  const expected=baseline.get(rel);
  if(expected===undefined&&target)fail('PUBLICATION_CONFLICT','An untracked destination already exists');
  if(expected!==undefined){if(!target)fail('PUBLICATION_CONFLICT','Original source disappeared');const old=await bytes(dest);if(sha(old.data)!==expected)fail('PUBLICATION_CONFLICT','Original source differs from copy baseline');}
  const sourceMode=b.stat.mode&0o777;const targetMode=target?target.mode&0o777:sourceMode;
  prepared.push({rel,dest,data:b.data,after:item.sha256,before:expected??null,mode:targetMode});
 }
 // Approval is separate from native preparation approval. The user sees the real destination,
 // exact filenames and immutable digests, rather than only the private working-copy path.
 const approved=await approve({root:original,files:prepared.map(p=>p.rel),digests:prepared.map(p=>({path:p.rel,sha256:p.after,preimage:p.before})),sessionId});
 if(approved!==true)fail('PUBLICATION_DECLINED','Original project publication was not authorized');
 // Every precondition is repeated after user approval: approval wait may be arbitrarily long.
 await pathState(original,{directory:true,owner:true});await pathState(privatePath,{directory:true,owner:true});
 for(const p of prepared){
  if(sha((await bytes(path.join(privatePath,p.rel))).data)!==p.after)fail('PUBLICATION_DIGEST','Verified private source changed after approval');
  const stat=await pathState(p.dest,{allowAbsent:true});
  if(p.before===null){if(stat)fail('PUBLICATION_CONFLICT','Destination appeared during approval');}
  else if(!stat||sha((await bytes(p.dest)).data)!==p.before)fail('PUBLICATION_CONFLICT','Original source changed during approval');
 }
 // Each approved attempt owns an atomically allocated directory, including retries
 // within the same session. Existing journals are never reused or overwritten.
 const journalRoot=path.join(storageRoot,'publication-journals');await ensurePublicationDirectory(journalRoot);
 const txn=await fs.mkdtemp(path.join(journalRoot,sessionId+'-'));await pathState(txn,{directory:true,owner:true});await syncDir(journalRoot);
 await newFile(path.join(txn,'intent.json'),Buffer.from(JSON.stringify({schema:'voder.original-publication.v1',sessionId,transactionId:path.basename(txn),source:privatePath,original,files:prepared.map(p=>({path:p.rel,before:p.before,after:p.after})),status:'INTENT'})+'\n'));
 await syncDir(txn);
 const staged=[];
 try{
  for(const p of prepared){
   const segments=p.rel.split('/').slice(0,-1);let d=original;
   for(const segment of segments){d=path.join(d,segment);await ensurePublicationDirectory(d);await syncDir(path.dirname(d));}
   const temp=path.join(path.dirname(p.dest),'.voder-publish-'+crypto.randomUUID());await newFile(temp,p.data,p.mode);staged.push({p,temp});
  }
  for(const {p,temp} of staged){
   if(p.before===null){if(await pathState(p.dest,{allowAbsent:true}))fail('PUBLICATION_CONFLICT','Target appeared before publication');try{await fs.link(temp,p.dest);}catch(e){if(e.code==='EEXIST')fail('PUBLICATION_CONFLICT','Target appeared during publication');throw e;}await fs.unlink(temp);}
   else{const prior=await bytes(p.dest);if(sha(prior.data)!==p.before)fail('PUBLICATION_CONFLICT','Source changed before publication');await fs.rename(temp,p.dest);}
   await syncDir(path.dirname(p.dest));
   if(sha((await bytes(p.dest)).data)!==p.after)fail('PUBLICATION_POSTCHECK','Destination digest differs after delivery');
  }
  await newFile(path.join(txn,'delivered.json'),Buffer.from(JSON.stringify({schema:'voder.original-publication.v1',sessionId,transactionId:path.basename(txn),status:'DELIVERED',files:prepared.map(p=>({path:p.rel,sha256:p.after}))})+'\n'));
  await syncDir(txn);
  return Object.freeze({status:'DELIVERED',root:original,files:prepared.map(p=>({path:p.rel,sha256:p.after}))});
 }catch(error){
  // Failure can be partial. Preserve the original error and intent for inspection;
  // never replay a write or delete another attempt's journal.
  try{await newFile(path.join(txn,'failed.json'),Buffer.from(JSON.stringify({schema:'voder.original-publication.v1',sessionId,transactionId:path.basename(txn),status:'FAILED',code:error.code??'PUBLICATION_ERROR'})+'\n'));await syncDir(txn);}catch{}
  throw error;
 }finally{for(const {temp} of staged)await fs.unlink(temp).catch(()=>{});} 
}
module.exports={publishOriginalProject};
