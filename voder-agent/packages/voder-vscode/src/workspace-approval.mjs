/** Host-owned, exact-scope approval. Model text and tool arguments cannot grant authority. */
import {randomUUID,createHash} from 'node:crypto';
const fail=(code,message)=>{throw Object.assign(Error(message),{code});};
export class WorkspaceApproval {
 constructor({mode='interactive',readOnly=false,emit=()=>{}}={}){
  if(!['interactive','preapproved'].includes(mode)||typeof readOnly!=='boolean')fail('APPROVAL_CONFIG','Invalid host approval policy');
  this.mode=mode;this.readOnly=readOnly;this.emit=emit;this.pending=null;this.closed=false;this.denied=false;this.requested=false;this.decision='NOT_REQUESTED';
 }
 async request(sessionId,scope){
  if(this.closed)fail('APPROVAL_CLOSED','This task cannot acquire new authority');
  if(this.denied)fail('WRITE_SCOPE_DECLINED','Write approval was declined for this task; do not ask again in the same task');
  if(this.readOnly)fail('TASK_READ_ONLY','The user explicitly requested a read-only task');
  this.requested=true;
  if(this.mode==='preapproved'){this.decision='PREAPPROVED';return true;}
  if(this.pending)fail('APPROVAL_BUSY','Another exact write scope is awaiting the user');
  const snapshot=structuredClone(scope),scopeDigest=createHash('sha256').update(JSON.stringify(snapshot)).digest('hex');
  const requestId=randomUUID();let resolve;
  const answer=new Promise(r=>{resolve=r;});
  this.pending=Object.freeze({sessionId,requestId,scopeDigest,scope:snapshot,resolve});this.decision='PENDING';
  try{this.emit({type:'task.workspace_approval_requested',sessionId,requestId,scopeDigest,root:snapshot.root,files:structuredClone(snapshot.files),checks:structuredClone(snapshot.checks)});}
  catch(e){this.pending=null;this.decision='DECLINED';resolve(false);throw e;}
  return answer;
 }
 decide(input){
  if(!input||typeof input!=='object'||Array.isArray(input)||Object.keys(input).sort().join(',')!=='approved,requestId,scopeDigest,sessionId'||typeof input.approved!=='boolean')fail('APPROVAL_ARGUMENTS','Closed host approval response required');
  const p=this.pending;
  if(this.closed||!p||input.sessionId!==p.sessionId||input.requestId!==p.requestId||input.scopeDigest!==p.scopeDigest)fail('APPROVAL_STALE','Approval does not match the pending live task and exact scope');
  this.pending=null;if(!input.approved)this.denied=true;this.decision=input.approved?'APPROVED':'DECLINED';p.resolve(input.approved);
  return {status:input.approved?'WRITE_SCOPE_APPROVED':'WRITE_SCOPE_DECLINED',requestId:p.requestId,scopeDigest:p.scopeDigest};
 }
 close(){this.closed=true;if(this.pending){const p=this.pending;this.pending=null;this.decision='CANCELLED';p.resolve(false);}}
 summary(){return {write_requested:this.requested,write_decision:this.decision,approval_mode:this.mode};}
}
