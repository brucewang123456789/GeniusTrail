/** Observe native/sandbox evidence; no model-generated diagnosis is an authority. */
import {createHash} from 'node:crypto';
const digest = value => createHash('sha256').update(value).digest('hex');
const CLASSES=[
  ['PATCH_FORMAT',/PATCH_FORMAT:|patch fragment without header|corrupt patch|malformed patch|invalid patch/i],
  ['PATCH_CONTEXT',/PATCH_CONTEXT:|patch does not apply|patch failed:|patch applicability/i],
  ['TIMEOUT',/\btimeout\b|timed? out|deadline exceeded/i],
  ['ENVIRONMENT',/module not found|cannot find module|no module named|dependency|no such file|command not found|not installed/i],
  ['COMPILE',/syntaxerror|compile error|compilation failed|ts\d{4}|rustc error|typeerror: .* is not assignable/i],
  ['ASSERTION',/assertionerror|assert\.equal|expected .*actual|test[_ ]fail(?:ed)?|\bassertion failed\b/i],
  ['RUNTIME',/typeerror|referenceerror|exception|traceback|panic|segmentation fault/i],
  ['POLICY',/permission denied|unauthorized|forbidden|scope denied|policy denied|credential/i],
  ['INFRASTRUCTURE',/503 service unavailable|connection refused|rate limit|network unreachable/i],
];
const NEXT=Object.freeze({PATCH_FORMAT:'USE_HOST_CONSTRUCTED_UNIFIED_DIFF',PATCH_CONTEXT:'REREAD_PINNED_SOURCE_AND_USE_EXACT_EDITS',TIMEOUT:'INSPECT_TIMEOUT_AND_BOUND_SCOPE',ENVIRONMENT:'INSPECT_LOCKFILE_AND_SANDBOX_ENV',COMPILE:'INSPECT_COMPILER_DIAGNOSTIC_AND_CONTRACT',ASSERTION:'ISOLATE_SMALLEST_REPRODUCTION',RUNTIME:'INSPECT_STACK_AND_RELATED_SOURCE',POLICY:'STOP_AND_REQUEST_AUTHORIZATION',INFRASTRUCTURE:'CHECK_PROVIDER_AND_RETRY_POLICY',UNKNOWN:'GATHER_DISCRIMINATING_EVIDENCE'});
export function classifyFailure(value){
  const raw=typeof value==='string'?value:'';
  for(const [label,re] of CLASSES)if(re.test(raw))return label;
  return 'UNKNOWN';
}
/** Event digests and types are preserved. Raw untrusted diagnostics never enter authority logs. */
export class DiagnosticJournal {
 constructor({maxIdenticalFailures=3}={}){
  if(!Number.isSafeInteger(maxIdenticalFailures)||maxIdenticalFailures<2||maxIdenticalFailures>10)throw new TypeError('Invalid stagnation threshold');
  this.maxIdenticalFailures=maxIdenticalFailures;this.histogram=new Map();this.count=0;this.blocked=false;
  this.last=null;this.lastSignature=null;
 }
 observeSandbox(receipt,{candidateDigest}={}){
  if(!receipt||!['PASS','FAIL','BLOCKED'].includes(receipt.status))throw new TypeError('Requires observed sandbox status');
  if(typeof candidateDigest!=='string'||!/^[a-f0-9]{64}$/.test(candidateDigest))throw new TypeError('Invalid candidate digest');
  const detail=typeof receipt.diagnostic==='string'?receipt.diagnostic.slice(0,4096):'';
  const failureClass=receipt.status==='PASS'?'NONE':classifyFailure(detail);
  const normalized=detail.toLowerCase().replace(/\b0x[a-f\d]+\b/g,'HEX').replace(/\b\d{4}-\d\d-\d\d\S*/g,'TIME').replace(/\b\d+\b/g,'N');
  const signature=digest(receipt.status+'|'+failureClass+'|'+normalized);
  const actionableSignature=digest(candidateDigest+'|'+signature);
  // Different bytes are a different attempt even when Git emits the same
  // coarse failure class. The scheduler still blocks duplicate byte digests,
  // and bounded tool/model budgets remain enforced.
  const repeat=receipt.status==='PASS'?0:(this.histogram.get(actionableSignature)??0)+1;
  if(receipt.status!=='PASS')this.histogram.set(actionableSignature,repeat);
  this.count++;
  if(receipt.status!=='PASS'&&repeat>=this.maxIdenticalFailures)this.blocked=true;
  this.lastSignature=signature;
  this.last=Object.freeze({source:receipt.backend==='host_patch_preflight'?'HOST_GIT_PREFLIGHT':'REAL_SANDBOX_RECEIPT',status:receipt.status,classification:failureClass,
    evidence_digest:typeof receipt.evidence_digest==='string'?receipt.evidence_digest:null,
    candidate_digest:candidateDigest,failure_signature:signature,identical_count:repeat,
    next_diagnostic:receipt.status==='PASS'?'VERIFY_BINDING_AND_COMMIT':NEXT[failureClass],
    causal_claim:'HYPOTHESIS_NOT_VERIFIED',stop:this.blocked});
  return this.last;
 }
 observeNative(receipt){
  if(!receipt||!['COMMITTED','DENIED','ROLLED_BACK','OBSERVED'].includes(receipt.status))throw new TypeError('Invalid authoritative result');
  const result=Object.freeze({source:'NATIVE_VGR_VERDICT',status:receipt.status,
    claim:receipt.status==='COMMITTED'?'COMMIT_REPORTED_BY_TRUSTED_BROKER':'NO_COMMIT',
    action_digest:receipt.commit_digest??null});
  return result;
 }
 nextModelStage(){this.blocked=false;}
 assertMayProceed(){if(this.blocked){const e=new Error('Unchanged verified failure signature reached the evidence budget; stop, do not repeat');e.code='DIAGNOSTIC_STAGNATION';throw e;}}
 snapshot(){return Object.freeze({observed_sandbox_checks:this.count,unique_failure_signatures:this.histogram.size,
  stalled:this.blocked,last_class:this.last?.classification??null,last_evidence_digest:this.last?.evidence_digest??null,
  identical_failure_count:this.last?.identical_count??0});}
}
/** RSI stays a quarantined method-candidate registry; it never patches live Pi or VGR. */
export class ImprovementCandidates {
 constructor({verifyEvaluation}={}){
  if(verifyEvaluation!==undefined&&typeof verifyEvaluation!=='function')throw new TypeError('Trusted evidence verifier must be a function');
  this.verifyEvaluation=verifyEvaluation;this.items=new Map();}
 propose({scope,method,methodVersion}={}){
  if(!['task-planning','repair-policy','skill'].includes(scope)||typeof method!=='string'||!method.trim()||Buffer.byteLength(method)>16384||
     typeof methodVersion!=='string'||!/^[a-zA-Z0-9._-]{1,64}$/.test(methodVersion))throw new TypeError('Invalid quarantined improvement');
  const id=digest(JSON.stringify([scope,methodVersion,method]));
  if(!this.items.has(id))this.items.set(id,{id,scope,methodVersion,candidate_digest:digest(method),evaluations:new Map(),authorized:false});
  return Object.freeze({id,status:'QUARANTINED',candidate_digest:digest(method)});
 }
 async recordEvaluation({id,taskId,baselinePass,candidatePass,baselineCost,candidateCost,regressionFree,receiptDigest}={}){
  const item=this.items.get(id);
  if(!item)throw new TypeError('Unknown improvement candidate');
  if(typeof taskId!=='string'||!/^[a-z0-9._-]{1,96}$/i.test(taskId)||item.evaluations.has(taskId)||
    typeof baselinePass!=='boolean'||typeof candidatePass!=='boolean'||typeof regressionFree!=='boolean'||
    ![baselineCost,candidateCost].every(v=>Number.isFinite(v)&&v>=0)||
    typeof receiptDigest!=='string'||!/^[a-f0-9]{64}$/.test(receiptDigest))throw new TypeError('Independent evaluation evidence invalid');
  if(!this.verifyEvaluation){const e=new Error('Independent evaluator not configured');e.code='INDEPENDENT_EVALUATOR_REQUIRED';throw e;}
  const claim=Object.freeze({id,taskId,baselinePass,candidatePass,baselineCost,candidateCost,regressionFree,receiptDigest});
  if(await this.verifyEvaluation(claim)!==true){const e=new Error('Independent evidence did not validate the evaluation');e.code='EVALUATION_NOT_VERIFIED';throw e;}
  item.evaluations.set(taskId,{baselinePass,candidatePass,baselineCost,candidateCost,regressionFree,receiptDigest});
  return this.assess(id);
 }
 assess(id){const item=this.items.get(id);if(!item)throw new TypeError('Unknown candidate');
  const reports=[...item.evaluations.values()];
  const safe=reports.every(x=>x.regressionFree&&(!x.baselinePass||x.candidatePass));
  const strict=reports.some(x=>(!x.baselinePass&&x.candidatePass)||(x.baselinePass&&x.candidatePass&&x.candidateCost<x.baselineCost));
  return Object.freeze({id,status:reports.length>=2&&safe&&strict?'ELIGIBLE_FOR_HUMAN_REVIEW':'QUARANTINED',
    independent_tasks:reports.length,no_regression_observed:safe,strict_improvement_observed:strict,
    live_code_modified:false,automatic_promotion:false});
 }
}
