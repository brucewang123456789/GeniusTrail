import {normalizedCandidate} from '../../voder-orchestration/src/index.mjs';
/** Host-configured, per-request isolated candidate verification. No VGR authority. */
import {spawn} from 'node:child_process';
import {createHash,randomUUID} from 'node:crypto';
import {readFile,writeFile,mkdir,mkdtemp,lstat,realpath} from 'node:fs/promises';
import {join,resolve,sep} from 'node:path';
import {VoderError} from '../../voder-agent-core/src/contracts.mjs';
const hash=x=>createHash('sha256').update(x).digest('hex');
const DEFAULT_MODAL_SETUP_BUDGET_MS=720_000;
const PATCH_PREFLIGHT_TIMEOUT_MS=20_000;
const PATCH_PREFLIGHT_OUTPUT_MAX=32_768;
function modalSetupBudget(){
 const raw=process.env.VODER_MODAL_SETUP_BUDGET_MS;
 const ms=raw===undefined ? DEFAULT_MODAL_SETUP_BUDGET_MS : Number(raw);
 if(!Number.isSafeInteger(ms)||ms<300_000||ms>1_800_000)fail('VERIFIER_POLICY','Invalid sandbox preparation budget');
 return ms;
}
function redactTransport(text){
 let output=String(text);
 for(const [key,value] of Object.entries(process.env)){
  if(/TOKEN|SECRET|PASSWORD|API_KEY|AUTHORIZATION/i.test(key)&&typeof value==='string'&&value.length>=6)
   output=output.split(value).join('[REDACTED]');
 }
 return output.slice(-8192);
}

const fail=(code,msg)=>{throw new VoderError(code,msg)};
// Exact private diagnostics; never relax the independent gateway's ordered scope guard.
export function scopeMismatchDetails({repositoryRoot,verifierRoot,policySchema,policyPaths,gatewayPaths,candidatePaths,batch}){
 const eq=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
 const repository_match=resolve(repositoryRoot)===resolve(verifierRoot);
 const policy_paths_match=eq(policyPaths,gatewayPaths);
 const candidate_paths_match=eq(candidatePaths,gatewayPaths);
 const candidate_shape_match=(batch===true)===(policySchema==='voder.gateway.policy.v2');
 const mismatch_at=candidate_paths_match?-1:Array.from({length:Math.max(candidatePaths.length,gatewayPaths.length)},(_,i)=>i).find(i=>candidatePaths[i]!==gatewayPaths[i]);
 return {repository_match,policy_paths_match,candidate_paths_match,candidate_shape_match,
  first_target_mismatch_index:mismatch_at===undefined?-1:mismatch_at,
  missing_targets:gatewayPaths.filter(p=>!candidatePaths.includes(p)),
  extra_targets:candidatePaths.filter(p=>!gatewayPaths.includes(p)),
  order_only_mismatch:!candidate_paths_match&&eq([...candidatePaths].sort(),[...gatewayPaths].sort())};
}
async function recordPrivateScopeDenial(gate,details,{sessionId,candidate_digest,policyPaths,gatewayPaths,candidatePaths,repositoryRoot,verifierRoot}){
 // The model/UI sees only VERIFIER_SCOPE; the private audit owns concrete paths.
 const folder=join(gate.attemptStore,'candidate-checks','scope-denials');
 await mkdir(folder,{recursive:true,mode:0o700});
 const id=randomUUID();
 const record={schema:'voder.private.scope-denial.v1',id,timestamp:new Date().toISOString(),
  session_id:String(sessionId??'host').slice(0,120),candidate_digest:typeof candidate_digest==='string'?candidate_digest.slice(0,128):null,
  details,policy_paths:policyPaths,gateway_paths:gatewayPaths,candidate_paths:candidatePaths,
  repository_hash:hash(resolve(repositoryRoot)),verifier_root_hash:hash(resolve(verifierRoot))};
 await writeFile(join(folder,id+'.private.json'),JSON.stringify(record)+'\n',{mode:0o600,flag:'wx'});
 return id;
}

async function execute(binary,args,{cwd,signal,timeoutMs=120000,limit=524288,input}={}){
 return new Promise((done,reject)=>{
  const child=spawn(binary,args,{cwd,env:process.env,stdio:[input===undefined?'ignore':'pipe','pipe','pipe'],shell:false});let output=Buffer.alloc(0),err=Buffer.alloc(0),timed=false,overflow=false;
  if(input!==undefined){child.stdin.on('error',e=>{if(e.code!=='EPIPE')reject(e);});child.stdin.end(input);}
  const abort=()=>child.kill('SIGKILL');signal?.addEventListener('abort',abort,{once:true});
  const timer=setTimeout(()=>{timed=true;abort()},timeoutMs);
  for(const [stream,kind] of [[child.stdout,0],[child.stderr,1]])stream.on('data',b=>{if(output.length+err.length+b.length>limit){overflow=true;abort();return;}if(kind)err=Buffer.concat([err,b]);else output=Buffer.concat([output,b]);});
  child.on('error',e=>{clearTimeout(timer);signal?.removeEventListener('abort',abort);reject(e)});
  child.on('close',(rc,term)=>{clearTimeout(timer);signal?.removeEventListener('abort',abort);done({rc,term,timed,overflow,stdout:output,stderr:err});});
 });
}
async function regular(path,max){const st=await lstat(path);if(!st.isFile()||st.isSymbolicLink()||st.size>max)fail('VERIFIER_UNSAFE_FILE','Unsafe verifier evidence');return readFile(path);}
// Git is a read-only syntax/applicability oracle here; Modal remains the only
// authority for independent execution, and Rust VGR remains the commit gate.
// This is intentionally scoped to tracked .patch/.diff delivery artifacts.
// The checked paths must be the *same* paths the validator will patch.
// In particular Git's `apply --check` can return 0 after silently skipping
// every patch from a repository subdirectory. Never treat that as approval.
function numstatPaths(buffer){
 const rows=buffer.toString('utf8').split('\0');if(rows.at(-1)==='')rows.pop();
 const paths=[];
 for(let i=0;i<rows.length;i++){
  const match=/^(?:\d+|-)\t(?:\d+|-)\t(.*)$/s.exec(rows[i]);
  if(!match)return null;
  let name=match[1];
  // Git uses an empty third field followed by old/new paths for renames.
  if(!name){if(i+2>=rows.length)return null;i++;const old=rows[i];i++;name=rows[i];if(!old||!name)return null;}
  if(!name||name.startsWith('/')||name.split('/').some(part=>part==='..'||part==='.'||!part))return null;
  paths.push(name);
 }
 return paths.length?paths:null;
}
export async function hostPatchPreflight({repo,dir,members,candidate_digest,signal,patchBaseDir='.'}){
 const patches=members.filter(m=>/\.(?:patch|diff)$/i.test(m.relative_path));
 if(!patches.length)return null;
 const checks=[];
 if(typeof patchBaseDir!=='string'||(patchBaseDir!=='.'&&(!patchBaseDir||patchBaseDir.split('/').some(p=>!p||p==='.'||p==='..'||! /^[A-Za-z0-9._-]+$/.test(p)))))fail('VERIFIER_PATCH_ROOT','Explicit relative patch root required');
 for(const member of patches){
  const parse=await execute('/usr/bin/git',['apply','--numstat','-z','-'],{cwd:repo,signal,input:member.new_content,timeoutMs:PATCH_PREFLIGHT_TIMEOUT_MS,limit:PATCH_PREFLIGHT_OUTPUT_MAX});
  const paths=parse.rc===0?numstatPaths(parse.stdout):null;
  checks.push({stage:'PARSE',file:member.relative_path,rc:parse.rc,timeout:parse.timed,overflow:parse.overflow,changes:paths?.length??0});
  if(parse.timed||parse.overflow||parse.rc===null)fail('VERIFIER_PREFLIGHT_INFRASTRUCTURE','Host Git syntax preflight could not complete');
  if(parse.rc!==0||!paths)return await preflightFailure(dir,candidate_digest,checks,'PATCH_FORMAT',parse.stderr.toString('utf8')||'No safely decoded changed paths');
  // The trusted adapter selects the source root. Directory existence never selects addressing.
  const args=['apply','--check','--verbose',...(patchBaseDir==='.'?[]:['--directory='+patchBaseDir]),'-'];
  const checked=await execute('/usr/bin/git',args,{cwd:repo,signal,input:member.new_content,timeoutMs:PATCH_PREFLIGHT_TIMEOUT_MS,limit:PATCH_PREFLIGHT_OUTPUT_MAX});
  const output=Buffer.concat([checked.stdout,Buffer.from('\n'),checked.stderr]).toString('utf8');
  const checkedPaths=(output.match(/(?:^|\n)Checking patch /g)||[]).length;
  const skipped=/(?:^|\n)Skipped patch /m.test(output);
  checks.push({stage:'APPLICABILITY',file:member.relative_path,context:patchBaseDir,rc:checked.rc,timeout:checked.timed,overflow:checked.overflow,expectedPaths:paths.length,checkedPaths,skipped});
  if(checked.timed||checked.overflow||checked.rc===null)fail('VERIFIER_PREFLIGHT_INFRASTRUCTURE','Host Git applicability preflight could not complete');
  if(checked.rc!==0||skipped||checkedPaths!==paths.length)
   return await preflightFailure(dir,candidate_digest,checks,'PATCH_CONTEXT',output||`Insufficient checked paths: ${checkedPaths}/${paths.length}`);
 }
 await writeFile(join(dir,'patch-preflight.private.json'),JSON.stringify({schema:'voder.private.patch-preflight.v1',status:'PASS',checks})+'\n',{mode:0o600,flag:'wx'});
 return null;
}
async function preflightFailure(dir,candidate_digest,checks,kind,error){
 const detail=redactTransport(error).slice(-2000);
 const diagnostic=`${kind}: ${detail || 'git apply rejected candidate'}; `+
  'repair the exact candidate; for .patch/.diff prefer unified_diff with pinned source files and exact edits so the host calculates hunk headers/counts';
 const evidence={schema:'voder.private.patch-preflight.v1',status:'FAIL',kind,checks,diagnostic};
 const bytes=Buffer.from(JSON.stringify(evidence)+'\n');
 await writeFile(join(dir,'patch-preflight.private.json'),bytes,{mode:0o600,flag:'wx'});
 return {status:'FAIL',backend:'host_patch_preflight',sandboxed:false,candidate_digest,checks:[{name:'git_patch_syntax_and_applicability',passed:false}],evidence_digest:hash(bytes),diagnostic};
}

export class HostCandidateVerifier {
 constructor({cwd,nativeGateway,timeoutMs=30000}={}){this.cwd=resolve(cwd);this.gate=nativeGateway;this.timeoutMs=timeoutMs;this.checkCounts=new Map();}
 getBudget(sessionId='host'){const used=this.checkCounts.get(sessionId)??0;return Object.freeze({limit:null,used,remaining:null});}
 async verifyCandidate(candidate,options={}){
  try{const id=options.sessionId??'host';const result=await this._verifyCandidate(candidate,options);this.checkCounts.set(id,(this.checkCounts.get(id)??0)+1);return result;}
  catch(e){if(['VERIFIER_INFRASTRUCTURE','VERIFIER_RECEIPT','VERIFIER_CLONE_FAILED','VERIFIER_BUDGET'].includes(e.code))queueMicrotask(()=>Promise.resolve(this.onInfrastructureError?.(options.sessionId)).catch(()=>{}));throw e;}
 }
 async _verifyCandidate(candidate,{candidate_digest,signal,onSandboxStart,sessionId}={}){
  if(signal?.aborted)fail('CANCELLED','Candidate check cancelled');
  candidate=normalizedCandidate(candidate);const members=candidate.patches??[candidate];
  const gate=this.gate;
  const policy=JSON.parse(await regular(gate.policyFile,16384));
  const expected=gate.targetPaths??[gate.targetPath];
  const policyPaths=policy.schema==='voder.gateway.policy.v2'?policy.target_paths:[policy.target_path];
  const candidatePaths=members.map(p=>p.relative_path);
  const mismatch=scopeMismatchDetails({repositoryRoot:policy.repository_path,verifierRoot:this.cwd,
   policySchema:policy.schema,policyPaths,gatewayPaths:expected,candidatePaths,batch:!!candidate.patches});
  if(!mismatch.repository_match||!mismatch.policy_paths_match||!mismatch.candidate_paths_match||!mismatch.candidate_shape_match){
   let diagnosticId=null;
   try{diagnosticId=await recordPrivateScopeDenial(gate,mismatch,{sessionId,candidate_digest,policyPaths,gatewayPaths:expected,candidatePaths,repositoryRoot:policy.repository_path,verifierRoot:this.cwd});}
   catch{/* Diagnostic IO must not change verifier acceptance or expose private paths. */}
   const error=new VoderError('VERIFIER_SCOPE','Candidate differs from complete operator-pinned policy; inspect private scope-denial diagnostics');
   // Classification only; source filenames and absolute paths stay private.
   error.scopeMismatchClass=(!mismatch.repository_match||!mismatch.policy_paths_match)?'HOST_POLICY_MISMATCH':'CANDIDATE_SCOPE_MISMATCH';
   error.scopeDiagnosticId=diagnosticId;
   throw error;
  }
  if(!['/usr/bin/','/bin/'].some(p=>policy.validator_program?.startsWith(p))||!Array.isArray(policy.validator_args)||policy.validator_args.length>16||policy.validator_args.some(x=>typeof x!=='string'||x.length>512))fail('VERIFIER_POLICY','Invalid validator policy');
  const digest=hash(`${policy.validator_program}\0v1\0[${policy.validator_args.map(x=>JSON.stringify(x)).join(', ')}]`);
  if(digest!==policy.validator_sha256)fail('VERIFIER_POLICY','Validator command digest mismatch');
  for(const member of members){const original=join(this.cwd,member.relative_path),canonical=await realpath(original);
   if(!canonical.startsWith(this.cwd+sep)||(await lstat(original)).isSymbolicLink()||hash(await readFile(original))!==member.old_sha256)fail('STALE_SOURCE','Candidate preimage differs');}
  const store=join(gate.attemptStore,'candidate-checks');await mkdir(store,{recursive:true,mode:0o700});
  const dir=await mkdtemp(join(store,'check-')),repo=join(dir,'repo');
  const clone=await execute('/usr/bin/git',['clone','--quiet','--no-hardlinks','--',this.cwd,repo],{cwd:dir,signal,timeoutMs:30000});
  if(clone.rc!==0||clone.timed||clone.overflow)fail('VERIFIER_CLONE_FAILED','Could not create independent candidate checkout');
  for(const member of members){if(hash(await readFile(join(repo,member.relative_path)))!==member.old_sha256)fail('STALE_SOURCE','Tracked source differs from proposed base');
   await writeFile(join(repo,member.relative_path),member.new_content,{mode:0o600});}
  const rejected=await hostPatchPreflight({repo,dir,members,candidate_digest,signal,patchBaseDir:gate.patchBaseDir??'.'});
  if(rejected)return rejected;
  const setupBudgetMs=modalSetupBudget();
  const request={workspace:repo,program:policy.validator_program,args:policy.validator_args,timeoutMs:this.timeoutMs,setupBudgetMs,memoryMiB:512,cpu:1,maxOutputBytes:262144,runId:'verify-'+randomUUID(),actionId:'candidate-'+randomUUID()};
  const paths=['request.json','receipt.json','stdout.raw','stderr.raw'].map(p=>join(dir,p));await writeFile(paths[0],JSON.stringify(request)+'\n',{mode:0o600,flag:'wx'});
  onSandboxStart?.({candidate_digest});
  const result=await execute(gate.nodeBinary,['--experimental-strip-types',gate.modalCli,...paths],{cwd:dir,signal,timeoutMs:this.timeoutMs+setupBudgetMs+180_000});
  // Keep stage evidence for successful transports too; silence in the model
  // stream cannot distinguish remote upload from execution or cleanup.
  const stages=result.stderr.toString('utf8').split(/\r?\n/).filter(line=>
   /^VODER_MODAL_(?:STAGE=|WAIT=|LIFECYCLE=|INPUT_PACK=)/.test(line));
  if(stages.length)await writeFile(join(dir,'transport-stages.private.log'),stages.join('\n')+'\n',{mode:0o600,flag:'wx'});
  if(result.rc!==0||result.timed||result.overflow||signal?.aborted){
   const diagnostic={schema:'voder.private.verifier-transport.v1',
    child_exit_code:result.rc,child_signal:result.term,
    host_timeout:result.timed,output_overflow:result.overflow,cancelled:!!signal?.aborted,
    task_timeout_ms:this.timeoutMs,prepare_budget_ms:setupBudgetMs,
    transport_stderr_tail:redactTransport(result.stderr.toString('utf8')),
    transport_stdout_tail:redactTransport(result.stdout.toString('utf8'))};
   await writeFile(join(dir,'transport-diagnostics.private.json'),JSON.stringify(diagnostic)+'\n',{mode:0o600,flag:'wx'});
   fail('VERIFIER_INFRASTRUCTURE',`Sandbox transport failed (rc=${result.rc ?? 'SIGNAL'}, timeout=${result.timed}, signal=${result.term ?? 'NONE'}); see private candidate-check diagnostics`);
  }
  const bytes=await regular(paths[1],1048576),receipt=JSON.parse(bytes),stdout=await regular(paths[2],262144),stderr=await regular(paths[3],262144);
  if(receipt.protocol!=='voder-sandbox-receipt-v1'||receipt.provider!=='modal'||receipt.status!=='EXECUTED'||receipt.runId!==request.runId||receipt.actionId!==request.actionId||receipt.image!==process.env.VODER_MODAL_IMAGE||receipt.stdoutSha256!==hash(stdout)||receipt.stderrSha256!==hash(stderr)||!Number.isInteger(receipt.exitCode)||receipt.exitCode<0||receipt.exitCode>255||typeof receipt.timeout!=='boolean'||typeof receipt.outputLimited!=='boolean')fail('VERIFIER_RECEIPT','Sandbox evidence mismatch');
  if([126,127].includes(receipt.exitCode)||receipt.timeout||receipt.outputLimited)fail('VERIFIER_INFRASTRUCTURE','Runtime unavailable, timed out or output limited');
  const passed=receipt.exitCode===0;
  return {status:passed?'PASS':'FAIL',backend:'modal',sandboxed:true,candidate_digest,checks:[{name:'host_pinned_validator',passed}],evidence_digest:hash(bytes),diagnostic:Buffer.concat([stdout,Buffer.from('\n'),stderr]).toString('utf8').slice(-4000)};
 }
}
