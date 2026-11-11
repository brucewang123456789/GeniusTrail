import {readBatchDiff} from './batch-delivery.mjs';
/** Voder A09: read-only Rust-authorized session diff; NOT a git diff or an approval grant. */
import {createHash} from 'node:crypto';
import {constants} from 'node:fs';
import {open,lstat,realpath,readFile} from 'node:fs/promises';
import {join,parse,resolve,relative,sep} from 'node:path';
import {TextDecoder} from 'node:util';
import {VoderError,isAuthorizedSourcePath} from '../../voder-agent-core/src/contracts.mjs';
const sha=b=>createHash('sha256').update(b).digest('hex');
const hex=/^[0-9a-f]{64}$/;
const safeId=/^action-[a-f0-9-]{36}$/i;
const absent=(id,reason)=>({schema:'voder.session.diff.v1',sessionId:id,status:'NOT_AVAILABLE',reason});
const deny=(code,msg)=>{throw new VoderError(code,msg)};
async function checkedPath(p,required='file'){
 const abs=resolve(p),top=parse(abs).root;let at=top;
 for(const segment of relative(top,abs).split(sep).filter(Boolean)){
  at=join(at,segment);const st=await lstat(at);
  if(st.isSymbolicLink())deny('DIFF_UNSAFE_PATH','Symbolic links cannot be trusted as evidence');
  if(at===abs && (required==='file'?!st.isFile():!st.isDirectory()))deny('DIFF_UNSAFE_TYPE','Unexpected evidence file type');
 }
 return abs;
}
async function bounded(path,limit=150000,allowEmpty=false){
 await checkedPath(path);
 const h=await open(path,constants.O_RDONLY|constants.O_NOFOLLOW);
 try{
  const st=await h.stat();if(!st.isFile()||(!allowEmpty&&st.size<1)||st.size>limit)deny('DIFF_UNSAFE_SIZE','Evidence exceeds read limit');
  const bytes=await h.readFile();if(bytes.length!==st.size)deny('DIFF_RACE','Evidence file changed during reading');return bytes;
 }finally{await h.close()}
}
async function jsonFile(p){let data;try{data=JSON.parse((await bounded(p,32768)).toString('utf8'))}catch(e){if(e.code?.startsWith('DIFF_'))throw e;deny('DIFF_PROOF_UNREADABLE','Missing or malformed native delivery evidence')}return data;}
function verifiedProof(proof,{id,action,run,commit}){
 return proof?.schema==='voder.verified-delivery.v1' && proof.status==='DELIVERED' &&
  proof.session_id===id && proof.action_id===action && proof.run_id===run && proof.commit_digest===commit &&
  hex.test(proof.journal_digest??'')&&hex.test(proof.before_sha256??'')&&hex.test(proof.after_sha256??'')&&
  isAuthorizedSourcePath(proof.relative_path);
}
/** No textual content is returned before Rust observer, filesystem proof and exact session are cross-checked. */
export async function readNativeSessionDiff({journal,id,attemptStore,policyFile,commitVerifier}={}){
 if(!journal||typeof journal.get!=='function')deny('DIFF_NO_JOURNAL','Private SDK journal required');
 const rec=await journal.get(id); // includes SHA/sequence/workspace checks
 if(!['DELIVERED'].includes(rec.status)||rec?.result?.verified!==true||rec?.result?.delivery_status!=='DELIVERED')
  return absent(id,'Task has no settled, verified native workspace delivery');
 if(!attemptStore||!policyFile||!commitVerifier||typeof commitVerifier.confirmCommit!=='function')
  return absent(id,'Independent Rust VGR observer and operator-pinned attempt store required');
 const actions=rec.native_actions;
 if(!Array.isArray(actions)||actions.length!==1||actions[0]?.schema!=='voder.dispatch.intent.v1'||
    actions[0].session_id!==id||actions[0].status!=='VERIFIED_NATIVE_COMMIT'||!safeId.test(actions[0].action_id))
  deny('DIFF_ACTION_IDENTITY','Only one journal-bound native committed action may produce a session diff');
 const act=actions[0].action_id,run=`run-${act}`,commit=rec?.result?.commit_digest;
 if(actions[0].run_id!==run||rec?.result?.run_id!==run||!hex.test(commit??''))deny('DIFF_COMMIT_IDENTITY','Journal commit identity is incomplete');
 const workspace=resolve(journal.workspace);if(workspace!==rec.workspace||await realpath(workspace)!==workspace)
  deny('DIFF_WORKSPACE','Canonical workspace does not match the journal');
 const attempt=join(resolve(attemptStore),'attempts',act,'verified-delivery');
 const [intent,delivered]=await Promise.all([jsonFile(join(attempt,'intent.json')),jsonFile(join(attempt,'delivered.json'))]);
 if(delivered?.schema==='voder.verified-batch-delivery.v1')return readBatchDiff({dir:attempt,workspace,id,action:act,run,commit,policyFile,commitVerifier});
 if(!verifiedProof(delivered,{id,action:act,run,commit})||intent?.status!=='INTENT'||
    Object.entries(delivered).some(([k,v])=>k!=='status'&&intent[k]!==v))
  deny('DIFF_PROOF_MISMATCH','Delivered and intent proofs do not match the native session');
 const policy=await jsonFile(policyFile);
 if(policy?.schema!=='voder.gateway.policy.v1'||resolve(policy.repository_path??'/')!==workspace||
    policy.target_path!==delivered.relative_path)
  deny('DIFF_GATEWAY_POLICY','Host-pinned VGR gateway policy does not authorize this source file');
 const target=resolve(workspace,delivered.relative_path);
 if(!target.startsWith(workspace+sep))deny('DIFF_PATH_ESCAPE','Delivery target escaped task workspace');
 const candidate=join(resolve(attemptStore),'attempts',act,'owned-worktrees','candidate',delivered.relative_path);
 const [before,after,candidateBytes]=await Promise.all([
  bounded(join(attempt,'before.bin'),150000,true),bounded(target),bounded(candidate)]);
 if(sha(before)!==delivered.before_sha256||sha(after)!==delivered.after_sha256||sha(candidateBytes)!==delivered.after_sha256)
  deny('DIFF_WORKTREE_CHANGED','Before image or live workspace bytes differ from recorded verified delivery');
 const receipt={status:'COMMITTED',authority:'VODER_NATIVE_VGR',session_id:id,action_id:act,run_id:run,
   commit_digest:commit,journal_digest:delivered.journal_digest};
 const authoritative=await commitVerifier.confirmCommit(receipt,{session_id:id,action_id:act,run_id:run});
 if(authoritative?.verified!==true||authoritative.authority!=='VODER_NATIVE_VGR_STATE'||
   authoritative.session_id!==id||authoritative.action_id!==act||authoritative.run_id!==run||authoritative.commit_digest!==commit)
  deny('DIFF_RUST_NOT_VERIFIED','Rust observer did not confirm exact final commit');
 if(sha(await bounded(target))!==delivered.after_sha256)
  deny('DIFF_WORKTREE_CHANGED','Workspace changed after independent Rust observation');
 let oldText,newText;
 try{const decoder=new TextDecoder('utf-8',{fatal:true});oldText=decoder.decode(before);newText=decoder.decode(after);
  if(oldText.includes('\0')||newText.includes('\0'))deny('DIFF_BINARY_UNSUPPORTED','Binary artifacts cannot be shown as text');}
 catch(e){if(e.code==='DIFF_BINARY_UNSUPPORTED')throw e;deny('DIFF_BINARY_UNSUPPORTED','Non-UTF8 artifacts are not displayed');}
 if(Buffer.byteLength(oldText)+Buffer.byteLength(newText)>255000)deny('DIFF_REVIEW_LIMIT','Verified file too large for bounded editor review');
 return {schema:'voder.session.diff.v1',sessionId:id,status:'VERIFIED_DELIVERED',authority:'VODER_NATIVE_VGR_STATE',
  action_id:act,run_id:run,commit_digest:commit,relative_path:delivered.relative_path,
  before_sha256:delivered.before_sha256,after_sha256:delivered.after_sha256,before:oldText,after:newText,
  note:'Read-only snapshot of native-delivered bytes; viewing does not authorize execution or new changes'};
}
