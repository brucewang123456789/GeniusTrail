/**
 * Voder v3: bounded evidence-guided tool control. It is a host mechanism, NOT
 * a model-visible instruction or a substitute for the Rust VGR commit verifier.
 */
import {createHash} from 'node:crypto';
import {VoderError,isAuthorizedSourcePath} from '../../voder-agent-core/src/contracts.mjs';

const hash = value => createHash('sha256').update(typeof value==='string'?value:JSON.stringify(value)).digest('hex');
const HEX=/^[a-f0-9]{64}$/;
function fail(code,message){throw new VoderError(code,message);}
export function normalizedCandidate(candidate){
 if(candidate&&Object.keys(candidate).join(',')==='patches'){
  if(!Array.isArray(candidate.patches)||candidate.patches.length<2||candidate.patches.length>16)fail('INVALID_CANDIDATE','Batch needs 2..16 existing files');
  const patches=candidate.patches.map(normalizedCandidate);
  if(patches.some(p=>p.patches)||patches.some((p,i)=>i&&patches[i-1].relative_path>=p.relative_path))fail('INVALID_CANDIDATE','Batch targets must be sorted, distinct and bounded');
  return Object.freeze({patches:Object.freeze(patches)});
 }

 if(!candidate || typeof candidate!=='object'||Array.isArray(candidate) ||
    Object.keys(candidate).sort().join(',')!=='new_content,old_sha256,relative_path' ||
    !isAuthorizedSourcePath(candidate.relative_path)||
    typeof candidate.old_sha256!=='string'||!HEX.test(candidate.old_sha256)||
    typeof candidate.new_content!=='string'||Buffer.byteLength(candidate.new_content,'utf8')<1)
    fail('INVALID_CANDIDATE','Invalid pinned candidate file, hash or content');
 return Object.freeze({...candidate});
}
export function candidateDigest(candidate){
 const c=normalizedCandidate(candidate);
 return hash(JSON.stringify(c.patches?c.patches.map(p=>[p.relative_path,p.old_sha256,p.new_content]):[c.relative_path,c.old_sha256,c.new_content]));
}
function optionsOf(opts={}){
 if(!opts||typeof opts!=='object'||Array.isArray(opts)||
    Object.keys(opts).some(k=>!['maxToolCalls','maxReadBytes','maxVerificationCalls','maxNoProgress'].includes(k)))
    fail('INVALID_ORCHESTRATION_POLICY','Unsupported orchestration configuration');
 const defaults={maxToolCalls:96,maxReadBytes:1024*1024,maxVerificationCalls:null,maxNoProgress:8};
 const limits={maxToolCalls:[4,256],maxReadBytes:[1024,16*1024*1024],maxVerificationCalls:[1,32],maxNoProgress:[2,32]};
 for(const [k,v] of Object.entries({...defaults,...opts})){const [min,max]=limits[k];
    if(k==='maxVerificationCalls'&&v===null)continue;
    if(!Number.isSafeInteger(v)||v<min||v>max)fail('INVALID_ORCHESTRATION_POLICY',`Invalid ${k}`);}
 return Object.freeze({...defaults,...opts});
}
export class EvidenceScheduler {
 constructor(opts={}){
  this.policy=optionsOf(opts);this.calls=0;this.bytes=0;this.verifications=0;this.modelStage=0;this.stageStartCalls=0;this.stageStartBytes=0;this.stageStartVerifications=0;
  this.evidenceRevision=0;this.noProgress=0;this.reads=new Map();this.inflightReads=new Set();
  this.probes=new Map();this.attemptedActions=new Set();this.observations=[];
  // One hard ledger; exploration cannot spend the verification/commit reserve.
  // Derived from the existing limits, including small host-configured budgets.
  this.finalizationReserve=Math.max(2,Math.min((this.policy.maxVerificationCalls??12)+1,Math.floor(this.policy.maxToolCalls/4)));
  this.phase='EXPLORING';this.phaseReason=null;this.explorationCalls=0;
  this.repairReadCredits=0;this.repairReads=0;this.repairDiagnostics=new Set();
  this.retrievalCalls=0;this.retrievalNoProgress=0;this.retrievalEvidence=new Set();this.retrievalInflight=new Set();
 }
 _remainingCalls(){return Math.max(0,this.policy.maxToolCalls-(this.calls-this.stageStartCalls));}
 _remainingBytes(){return Math.max(0,this.policy.maxReadBytes-(this.bytes-this.stageStartBytes));}
 // Model stages are finite (one stage for each configured, available model).
 // All PASS/FAIL receipts, candidate hashes, VGR actions and original source pins survive.
 advanceModel(){
  this.modelStage++;this.stageStartCalls=this.calls;this.stageStartBytes=this.bytes;this.stageStartVerifications=this.verifications;
  this.phase='EXPLORING';this.phaseReason=null;this.noProgress=0;this.repairReadCredits=0;this.repairReads=0;this.repairDiagnostics.clear();
  return this.snapshot();
 }
 finalize(reason){if(this.phase==='EXPLORING'){this.phase='FINALIZING';this.phaseReason=reason;}}
 bindVerificationLimit(limit){
  if(this.calls||!Number.isSafeInteger(limit)||limit<1||limit>32)fail('INVALID_ORCHESTRATION_POLICY','Bind host verification capacity before any tool call');
  this.policy=optionsOf({...this.policy,maxVerificationCalls:Math.min(limit,this.policy.maxVerificationCalls??limit)});
  this.finalizationReserve=Math.max(2,Math.min((this.policy.maxVerificationCalls??12)+1,Math.floor(this.policy.maxToolCalls/4)));
 }
 _call(kind){
  if(this._remainingCalls()<=0)fail('TOOL_BUDGET_EXHAUSTED','Tool-call budget exhausted, no silent replay');
  if(kind==='read'){
   if(this.noProgress>=this.policy.maxNoProgress)this.finalize('NO_EVIDENCE_PROGRESS');
   if(this._remainingCalls()<=this.finalizationReserve)this.finalize('READ_BUDGET_RESERVED');
   if(this.phase==='FINALIZING'){
    // Only new, completed sandbox FAIL evidence can fund a focused repair read.
    // At most TWO per task, charged to the same ledger, leaving verify + commit.
    if(this.repairReadCredits>0&&this._remainingCalls()>2){this.repairReadCredits--;this.repairReads++;}
    else fail(this.phaseReason,'Exploration ended. Use existing evidence to verify a candidate, then allow the host to prioritize native submission of the exact PASS digest; otherwise report the blocker. Do not repeat reads.');
   }else this.explorationCalls++;
  }
  // Even the last sandbox attempt must leave one host-controlled native action available.
  if(kind==='verify'&&this._remainingCalls()<=1){
   this.finalize('COMMIT_BUDGET_RESERVED');
   fail('COMMIT_BUDGET_RESERVED','Only native submission capacity remains; commit an existing PASS digest or report the blocker');
  }
  this.calls++;
 }
 async inspectRead(tool,args,run){
  if(tool==='voder_ext_search_evidence'&&['web','research','fetch','read'].includes(args?.mode))return this.inspectRetrieval(tool,args,run);
  this._call('read');
  const key=hash(JSON.stringify([tool,args]));
  if(this.inflightReads.has(key)||this.reads.get(key)===this.evidenceRevision) {
    this.noProgress++;fail('DUPLICATE_NO_PROGRESS','No new evidence since this identical repository read');
  }
  this.inflightReads.add(key);
  try{
   const result=await run();
   const bytes=Buffer.byteLength(JSON.stringify(result),'utf8');
   if(bytes>this._remainingBytes()){this.finalize('EVIDENCE_BYTE_BUDGET');fail('EVIDENCE_BYTE_BUDGET','Read evidence exceeds bounded context budget; use existing evidence to verify, or report the blocker (nothing truncated)');}
   this.bytes+=bytes;
   const digest=hash(result);
   if(!this.observations.some(x=>x.digest===digest&&x.tool===tool)){this.evidenceRevision++;this.noProgress=0;}
   else this.noProgress++;
   this.reads.set(key,this.evidenceRevision);
   this.observations.push(Object.freeze({tool,digest,bytes}));
   return result;
  }catch(e){if(!['EVIDENCE_BYTE_BUDGET'].includes(e?.code))this.noProgress++;throw e;}
  finally{this.inflightReads.delete(key);}
 }
 async inspectRetrieval(tool,args,run){
  // This read-only lane cannot spend or replenish native verification/commit capacity.
  const key=hash([tool,args]);
  if(this.retrievalInflight.has(key))fail('DUPLICATE_NO_PROGRESS','Identical retrieval is already in progress');
  if(this.retrievalNoProgress>=this.policy.maxNoProgress)fail('RETRIEVAL_NO_PROGRESS','External retrieval produced no new evidence repeatedly. Finish using supported findings and state the remaining gaps.');
  this.retrievalInflight.add(key);this.retrievalCalls++;
  try{
   const result=await run(),data=result?.data??result;
   const rows=Array.isArray(data?.results)?data.results:[];let novel=0;
   for(const r of rows){
    const signature=hash([r.url,r.content_hash??'',r.passage?.start??0,r.excerpt??'']);
    if(!this.retrievalEvidence.has(signature)){this.retrievalEvidence.add(signature);novel++;}
   }
   if(novel){this.retrievalNoProgress=0;this.evidenceRevision++;}else this.retrievalNoProgress++;
   while(this.retrievalEvidence.size>2048)this.retrievalEvidence.delete(this.retrievalEvidence.values().next().value);
   return result;
  }catch(e){if(!['CANCELLED','WEB_FORBIDDEN_BY_TASK','WEB_DISABLED','WEB_QUERY_BUDGET','INVALID_SEARCH_QUERY'].includes(e?.code))this.retrievalNoProgress++;throw e;}
  finally{this.retrievalInflight.delete(key);}
 }
 async inspectCandidate(candidate,{run,sourceDigest}){
  if(this.policy.maxVerificationCalls!==null&&this.verifications-this.stageStartVerifications>=this.policy.maxVerificationCalls){this.finalize('VERIFY_BUDGET_EXHAUSTED');fail('VERIFY_BUDGET_EXHAUSTED','Explicit operator verification capacity exhausted.');}
  this._call('verify');
  const c=normalizedCandidate(candidate);const key=candidateDigest(c);
  if(this.probes.has(key)){this.noProgress++;fail('DUPLICATE_CANDIDATE','Identical candidate was already tested; use its previous evidence to change strategy');}
  if(this.policy.maxVerificationCalls!==null&&this.verifications-this.stageStartVerifications>=this.policy.maxVerificationCalls){this.finalize('VERIFY_BUDGET_EXHAUSTED');fail('VERIFY_BUDGET_EXHAUSTED','Explicit operator verification capacity exhausted.');}
  // Reserve before asynchronous path inspection; an unknown remote outcome must
  // never be automatically replayed, even when no verified receipt was returned.
  this.probes.set(key,Object.freeze({status:'OUTCOME_UNKNOWN'}));
  let reply;
  try{
   for(const p of c.patches??[c])if(await sourceDigest(p.relative_path)!==p.old_sha256)fail('STALE_SOURCE','Candidate base digest differs from current repository source');
   reply=await run(c,key);
  }catch(error){
   if(error?.code==='STALE_SOURCE'&&this.probes.get(key)?.status==='OUTCOME_UNKNOWN')this.probes.delete(key);
   throw error;
  }

  if(!reply||typeof reply!=='object'||!['PASS','FAIL','BLOCKED'].includes(reply.status)||
     typeof reply.backend!=='string'||!reply.backend.trim()||reply.backend.length>96||
     (reply.status==='PASS'&&(!Array.isArray(reply.checks)||reply.checks.length<1||
       reply.checks.length>64||reply.checks.some(x=>typeof x?.name!=='string'||!x.name||x.name.length>120||x.passed!==true)||
       !HEX.test(reply.evidence_digest??'')||reply.sandboxed!==true)))
    fail('INVALID_SANDBOX_RECEIPT','Sandbox adapter failed to provide a valid bounded verification receipt');
  this.verifications++;
  const display={status:reply.status,backend:reply.backend,passed_checks:reply.status==='PASS'?reply.checks.length:0,
     evidence_digest:reply.status==='PASS'?reply.evidence_digest:undefined,
     diagnostic:typeof reply.diagnostic==='string'?reply.diagnostic.slice(0,4096):undefined};
  this.probes.set(key,Object.freeze({status:reply.status,source:c.old_sha256,receipt:display}));
  const observation=hash(display);
  if(reply.status==='FAIL'&&!this.repairDiagnostics.has(observation)&&this.repairReads+this.repairReadCredits<2&&this._remainingBytes()>0){
   this.repairDiagnostics.add(observation);this.repairReadCredits++;
  }
  if(reply.status==='PASS'){this.evidenceRevision++;this.noProgress=0;}else {
   this.evidenceRevision++;this.noProgress++;
  }
  this.observations.push(Object.freeze({tool:'voder_verify_candidate',digest:observation,bytes:Buffer.byteLength(JSON.stringify(display))}));
  return display;
 }
 async inspectAction(candidate,run, {verificationRequired=false,sourceDigest}={}){
  this._call('action');
  const c=normalizedCandidate(candidate);const key=candidateDigest(c);
  if(this.attemptedActions.has(key)){this.noProgress++;fail('DUPLICATE_ACTION','Identical VGR submission has already been attempted');}
  if(verificationRequired){
   if(this.probes.get(key)?.status!=='PASS')fail('VERIFY_BEFORE_COMMIT','Sandbox PASS for exact proposed bytes is required before native commit');
   for(const p of c.patches??[c])if(await sourceDigest(p.relative_path)!==p.old_sha256)fail('STALE_SOURCE','Pinned source changed after sandbox verification');
  }
  this.attemptedActions.add(key);
  // Never repeat a VGR action even after timeout/unknown outcome; this is separate from Pi's tool policy.
  const outcome=await run();
  if(outcome?.status==='COMMITTED'){this.evidenceRevision++;this.noProgress=0;}
  else this.noProgress++;
  this.observations.push(Object.freeze({tool:'voder_authorized_action',digest:hash([key,outcome?.status]),bytes:0}));
  return outcome;
 }
 control(){return Object.freeze({...this.snapshot(),verified_candidates:[...this.probes].filter(([key,value])=>value.status==='PASS'&&!this.attemptedActions.has(key)).map(([key])=>key).slice(-32),next_step:this.phase==='EXPLORING'
  ?'Inspect only what is needed; verify a concrete candidate before exhausting exploration. Reserve is not permission to write.'
  :this.repairReadCredits>0&&this._remainingCalls()>2
   ?'A new sandbox FAIL permits a focused repair read using its diagnostic. Use new evidence to repair a candidate, never repeat an identical digest; native authorization never resets.'
   :'No further exploration in this model stage. Verify a materially revised candidate, let the host submit its exact PASS digest, or report a blocker.'});}
 snapshot(){
  if(this.noProgress>=this.policy.maxNoProgress)this.finalize('NO_EVIDENCE_PROGRESS');
  if(this._remainingCalls()<=this.finalizationReserve)this.finalize('READ_BUDGET_RESERVED');
  if(this.policy.maxVerificationCalls!==null&&this.verifications-this.stageStartVerifications>=this.policy.maxVerificationCalls)this.finalize('VERIFY_BUDGET_EXHAUSTED');
  return Object.freeze({tool_calls:this.calls,retrieval_calls:this.retrievalCalls,retrieval_no_progress:this.retrievalNoProgress,read_bytes:this.bytes,verification_calls:this.verifications,
  phase:this.phase,phase_reason:this.phaseReason,model_stage:this.modelStage,exploration_calls:this.explorationCalls,
  remaining_tool_calls:this._remainingCalls(),reserved_finalization_calls:this.finalizationReserve,
  remaining_verification_calls:this.policy.maxVerificationCalls===null?null:Math.max(0,this.policy.maxVerificationCalls-(this.verifications-this.stageStartVerifications)),
  repair_read_credits:this.repairReadCredits,repair_reads:this.repairReads,
  evidence_revisions:this.evidenceRevision,no_progress:this.noProgress,
  positive_sandbox_checks:[...this.probes.values()].filter(x=>x.status==='PASS').length,
  attempted_native_actions:this.attemptedActions.size});}
}

/** Verifier is a trusted host-configured interface; it NEVER executes a model-provided command. */
export function assertCandidateVerifier(verifier){
 if(verifier===undefined)return;
 if(!verifier||typeof verifier!=='object'||typeof verifier.verifyCandidate!=='function')
  fail('INVALID_SANDBOX_ADAPTER','A host-owned verifyCandidate() sandbox adapter is required');
}
