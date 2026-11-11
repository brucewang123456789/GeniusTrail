import {WorkspaceApproval} from './workspace-approval.mjs';
/** Ordinary task host: one model session, on-demand scope, explicit final commit. */
import {createRequire} from 'node:module';
import {pathToFileURL,fileURLToPath} from 'node:url';
import {resolve,join,dirname} from 'node:path';
import {readFile,writeFile,mkdir,lstat,link,unlink} from 'node:fs/promises';
import {createHash,randomUUID} from 'node:crypto';
import {renderSvg} from './svg-preview.mjs';
import {inspectHtml,validateBrowserInput} from './html-browser.mjs';
const require=createRequire(import.meta.url);
const {readLaunch,real}=require('./general-workspace.cjs');
const {writeRequest}=require('./entry-orchestrator.cjs');
const {planTask,createTask,validateTask,snapshot}=require('./task-workspace.cjs');
const sha=b=>createHash('sha256').update(b).digest('hex');
export function evaluateVisualCommitReadiness(candidate,targets,renderAttempts,renders){
 const members=candidate?.patches??[candidate];
 if(!Array.isArray(members)||members.some(x=>typeof x?.relative_path!=='string'||typeof x?.new_content!=='string'))return {status:'WAIT',code:'CANDIDATE_FORMAT'};
 if(JSON.stringify(members.map(x=>x.relative_path))!==JSON.stringify(targets))return {status:'WAIT',code:'SCOPE_CHANGED'};
 const visual=members.filter(x=>/\.(?:html|svg)$/.test(x.relative_path));
 if(!visual.length)return {status:'EXPLICIT_ONLY',code:'NON_VISUAL_TASK_UNCHANGED'};
 for(const item of visual){
  const digest=sha(item.new_content),latest=renderAttempts.get(item.relative_path),row=renders.get(digest);
  if(latest?.status!=='RENDERED'||latest.source_sha256!==digest||row?.path!==item.relative_path||!row.exposed||row.review?.visual_decision!=='ACCEPT')return {status:'WAIT',code:'VISUAL_REVIEW_NOT_ACCEPTED'};
  if(item.relative_path.endsWith('.html')&&row.record.kind!=='HTML_BROWSER')return {status:'WAIT',code:'HTML_BROWSER_EVIDENCE_REQUIRED'};
 }
 return {status:'READY'};
}

const fail=(code,m)=>{throw Object.assign(Error(m),{code});};
const hostScope=(files,checks=[])=>({action:'execute',answer:'Task-selected source scope; not a model planning response.',files,checks});
/** Only gate verification when the exact candidate has an exposed, unaccepted
 * rendered image. Never block first-time verification or nonvisual tasks. */
export function visualReviewBeforeVerification(candidate,renderAttempts,renders){
 const members=candidate?.patches??[candidate];
 if(!Array.isArray(members))return null;
 for(const part of members){
  if(typeof part?.relative_path!=='string'||typeof part?.new_content!=='string'||!/\.(?:html|svg)$/.test(part.relative_path))continue;
  const digest=sha(part.new_content),latest=renderAttempts.get(part.relative_path),row=renders.get(digest);
  if(latest?.status==='RENDERED'&&latest.source_sha256===digest&&row?.path===part.relative_path&&row.exposed&&row.review?.visual_decision!=='ACCEPT'){
   return {code:'VISUAL_REVIEW_PENDING',path:part.relative_path,source_sha256:digest,decision:row.review?.visual_decision??'NOT_RECORDED'};
  }
 }
 return null;
}
/** Preserve the verifier's denial; make the authorized scope intelligible. */
export function verifierScopeGuidance(candidate,targets){
 const members=candidate?.patches??[candidate];
 const paths=Array.isArray(members)?members.map(x=>x?.relative_path).filter(x=>typeof x==='string'):[];
 const unique=paths.length===new Set(paths).size;
 const expected=Array.isArray(targets)?targets:[];
 const missing=expected.filter(x=>!paths.includes(x)),extra=paths.filter(x=>!expected.includes(x));
 const msg=`Verifier rejected candidate file scope. Approved scope: ${JSON.stringify(expected)}; proposed paths: ${JSON.stringify(paths)}.`;
 return (msg+` Missing: ${JSON.stringify(missing)}; outside scope: ${JSON.stringify(extra)}; unique paths: ${unique}. Use the exact approved relative paths and current preimage hashes, or call voder_task_prepare for a genuine additional source file and obtain user approval. Never bypass the verifier.`).slice(0,460);
}

/**
 * Remote browser replay policy. Only exact, canonicalized requests are compared.
 * Browser inspection is observational, not VGR verification or visual quality PASS.
 * A model can make progress with changed HTML, probe, or checks without any global
 * browser-count limit; repeats without new evidence return actionable feedback.
 */
export function createBrowserRepairGate(){
 const entries=new Map(),maxEntries=64;
 const put=(key,row)=>{entries.delete(key);entries.set(key,row);while(entries.size>maxEntries){const victim=[...entries].find(([,entry])=>entry.state!=='RUNNING');if(!victim)break;entries.delete(victim[0]);}};
 const advice='Use the previous browser frames and check results. If the scene has a defect, revise the HTML content before inspecting again; if the checks did not cover the requirement, use a justified different probe or check. If the evidence is sufficient, record the review for the final source hash, then verify the candidate and commit only after independent PASS. Do not repeat the identical browser request.';
 return Object.freeze({
  before(key){
   const row=entries.get(key);
   if(!row)return null;
   if(row.state==='RUNNING')return {code:'HTML_BROWSER_ALREADY_RUNNING',reason:'The identical browser inspection is still running. Consume its result instead of starting another remote sandbox.',terminal:false};
   if(row.state==='SUCCEEDED'){
    row.ignored=(row.ignored??0)+1;
    return {code:'HTML_BROWSER_REPAIR_REQUIRED',reason:'Identical HTML, checks and browser probe were already inspected; repeating them cannot establish new evidence. '+advice,terminal:row.ignored>=2};
   }
   if(row.state==='FAILED'&&row.runs>=2){
    row.ignored=(row.ignored??0)+1;
    return {code:'HTML_BROWSER_RETRY_UNPRODUCTIVE',reason:'The identical remote browser input failed twice. Do not assume the HTML is defective: diagnose the upstream failure, adjust an evidence-bearing check if appropriate, or report an unavailable browser service. '+advice,terminal:row.ignored>=2};
   }
   return null; // One transient remote failure may be retried exactly once.
  },
  start(key){const previous=entries.get(key);put(key,{state:'RUNNING',runs:(previous?.runs??0)+1,ignored:0});},
  finish(key,success){const row=entries.get(key);if(!row||row.state!=='RUNNING')throw Error('BROWSER_REPAIR_GATE_STATE');row.state=success?'SUCCEEDED':'FAILED';put(key,row);},
  inspect(key){const row=entries.get(key);return row?{state:row.state,runs:row.runs,ignored:row.ignored}:null;}
 });
}

/**
 * Hash-scoped evidence handoff. A real visual decision is mandatory, but the model
 * may explicitly request new evidence for a *different inspection dimension*.
 * Browser inputs, user scope, independent verification and VGR stay unchanged.
 */
export function createVisualDecisionGate(){
 const byPath=new Map();
 const advice='A successful browser check is evidence, not final visual acceptance. Inspect the returned images and call voder_task_review_render with ACCEPT, REPAIR_REQUIRED, or NEEDS_MORE_EVIDENCE. For NEEDS_MORE_EVIDENCE, explain a concrete unmet inspection requirement in limitations and make the next browser request test a different probe, check or playback duration. Changing preview_path alone is not new evidence.';
 return Object.freeze({
  before(path,digest,coverage){
   const row=byPath.get(path);
   if(!row)return null;
   // A new HTML digest is not permission to bypass the outstanding visual decision.
   // Once the model has honestly reviewed the last evidence, revised bytes can be inspected.
   if(row.digest!==digest){
    // The previous browser evidence belongs to superseded source bytes. Requiring
    // a review of that OLD build cannot improve the FINAL candidate's quality.
    // The new build still needs its own real frames, ACCEPT and VGR PASS.
    if(row.pending)return {code:'HTML_BROWSER_REVIEW_PENDING',reason:'The earlier browser run is still in progress; consume its result before launching another.',terminal:false};
    return null;
   }
   if(row.pending)return {code:'HTML_BROWSER_SAME_SOURCE_BUSY',reason:'A further inspection of these HTML bytes is already in progress; consume its result before another request.',terminal:false};
   // Novel evidence (new controls/checks/playback) is useful even if the model
   // has not filed a bureaucratic NEEDS_MORE_EVIDENCE review of old frames.
   if(coverage&&!row.coverages.has(coverage)&&row.decision!=='ACCEPT'&&row.decision!=='REPAIR_REQUIRED')return null;
   row.ignored++;
   const phase=row.decision;
   const code=phase==='REPAIR_REQUIRED'?'HTML_VISUAL_REPAIR_NOT_APPLIED':phase==='ACCEPT'?'HTML_VISUAL_ACCEPTED_NO_REINSPECT':phase==='NEEDS_MORE_EVIDENCE'?'HTML_BROWSER_NO_NEW_COVERAGE':'HTML_VISUAL_DECISION_REQUIRED';
   const reason=phase==='REPAIR_REQUIRED'?'Visual review reported a defect. Modify the actual HTML before inspecting again.':phase==='ACCEPT'?'Final visual review already accepted these bytes. Continue exact-candidate verification and Host commit.':phase==='NEEDS_MORE_EVIDENCE'?'The requested inspection covers a dimension already inspected for these HTML bytes. Test a genuinely different probe/check/playback condition and use the evidence, or make a final ACCEPT/REPAIR_REQUIRED decision.':advice;
   return {code,reason,terminal:row.ignored>=2};
  },
  starting(path,digest,coverage){const row=byPath.get(path);if(row&&row.digest===digest){if(row.pending)throw Error('VISUAL_EVIDENCE_CONCURRENT');row.pending=coverage;}},
  failed(path,digest,coverage){const row=byPath.get(path);if(row&&row.digest===digest&&row.pending===coverage)row.pending=null;},
  inspected(path,digest,coverage){
   const row=byPath.get(path);
   if(row&&row.digest===digest){row.coverages.add(coverage);while(row.coverages.size>128)row.coverages.delete(row.coverages.values().next().value);row.decision=null;row.ignored=0;row.pending=null;return;}
   byPath.set(path,{digest,decision:null,ignored:0,pending:null,coverages:new Set([coverage]),missingReasons:new Map()});
  },
  reviewed(path,digest,decision,limitations=''){
   const row=byPath.get(path);
   if(!row||row.digest!==digest||!['ACCEPT','REPAIR_REQUIRED','NEEDS_MORE_EVIDENCE'].includes(decision))return false;
   if(decision==='NEEDS_MORE_EVIDENCE'){
    const reason=limitations.trim().replace(/\s+/g,' ').toLowerCase();
    // Only an actual unmet requirement matters; word count and repeated
    // wording are not independent quality criteria. Novel browser coverage is
    // already enforced by the browser repair/evidence gates.
    if(!reason)return false;
    row.missingReasons.set(reason,(row.missingReasons.get(reason)??0)+1);
   }
   row.decision=decision;return true;
  },
  duplicateReview(path,digest){const row=byPath.get(path);if(!row||row.digest!==digest)return {terminal:false};row.ignored++;return {terminal:row.ignored>=2};},
  snapshot(path){const row=byPath.get(path);return row?{digest:row.digest,decision:row.decision,ignored:row.ignored,coverage_count:row.coverages.size}:null;}
 });
}

/** Replay exact browser frames once or twice without another paid sandbox.
 * Only an unchanged request and the same still-current evidence identity qualify.
 * This is evidence transport recovery, never quality ACCEPT or a VGR bypass. */
export function createVisualEvidenceReplay(){
 const items=new Map(),maxItems=8;
 return Object.freeze({
  remember(key,digest,row){
   items.delete(key);items.set(key,{digest,row,replays:0});
   while(items.size>maxItems)items.delete(items.keys().next().value);
  },
  request(key,digest,currentRow){
   const entry=items.get(key);
   if(!entry||entry.digest!==digest||entry.row!==currentRow||currentRow?.review?.visual_decision==='ACCEPT')return null;
   entry.replays++;
   if(entry.replays<=2)return {status:'REPLAY_FRAMES',row:entry.row,replays:entry.replays};
   return {status:'REVIEW_REQUIRED',replays:entry.replays,terminal:entry.replays>=4};
  },
  inspect(key){const entry=items.get(key);return entry?{digest:entry.digest,replays:entry.replays}:null;}
 });
}

export async function makeGeneralControl({cwd,sessionDir,gate,agentRoot,modelHost,browserDriver}={}){
 const authorization=await readLaunch(gate);
 if(authorization.root!==cwd||authorization.agentRoot!==agentRoot)fail('GENERAL_SESSION_IDENTITY','Workspace or agent root differs from launch authorization');
 const load=p=>import(pathToFileURL(join(agentRoot,p)).href);
 const {createTrace}=await load('packages/voder-agent-core/src/private-trace.mjs');const trace=createTrace(cwd);trace.write('launch_ready');
 const [{VoderSDK,VoderControlPlane,VoderJournal},{HostTaskWorkspace},{HostCandidateVerifier},{TrustedVgrGatewayTransport,GatewayCommitVerifier},{BatchWorkspaceDelivery},{VerifiedWorkspaceDelivery,regularBytes}]=await Promise.all([
  load('packages/voder-sdk/src/index.mjs'),load('packages/voder-agent-core/src/task-workspace.mjs'),load('packages/voder-sdk/src/candidate-verifier.mjs'),load('packages/voder-native-bridge/src/trusted-gateway.mjs'),load('packages/voder-sdk/src/batch-delivery.mjs'),load('packages/voder-sdk/src/verified-delivery.mjs')]);
 let activeSessionId=null;const approvalListeners=new Map();
 const approval=new WorkspaceApproval({mode:authorization.approvalMode??'preapproved',readOnly:authorization.readOnly??false,emit:event=>{trace.write('approval_emitted',{session_id:event.sessionId});for(const fn of approvalListeners.get(event.sessionId)??[])fn(event);}});
 let current=null,preparing=false,scopeFrozen=false,finalDelivery=null;const renders=new Map(),renderAttempts=new Map(),artifacts=[];let renderCalls=0,browserExecutions=0,creationIntent=false,blocker=null;const browserAttemptHistory=[];const browserRepairGate=createBrowserRepairGate(),visualDecisionGate=createVisualDecisionGate(),visualEvidenceReplay=createVisualEvidenceReplay();
 async function loadScope(project){
  const cfg=JSON.parse(await readFile(project.profile,'utf8')),transport=new TrustedVgrGatewayTransport(cfg);await transport.preflight();
  return {project,cfg,transport,verifier:new HostCandidateVerifier({cwd,nativeGateway:cfg,timeoutMs:180000}),commitVerifier:new GatewayCommitVerifier({daemonBinary:cfg.daemonBinary,attemptStore:cfg.attemptStore}),delivery:new (cfg.targetPaths?BatchWorkspaceDelivery:VerifiedWorkspaceDelivery)({cwd,attemptStore:cfg.attemptStore,policyFile:cfg.policyFile})};
 }
 const activeScopePath=join(dirname(gate),'active-scope.json'),reservationPath=join(dirname(gate),'session-reservation.json');
 let restored=false;
 try{await real(reservationPath,{file:true,privateFile:true});restored=true;}catch(e){if(e.code!=='ENOENT')throw e;}
 if(restored){
  scopeFrozen=true;
  try{await real(activeScopePath,{file:true,privateFile:true});const r=JSON.parse(await readFile(activeScopePath,'utf8'));const project=await validateTask(r.recordPath,authorization.storageRoot);if(project.root!==cwd||project.profile!==r.profile)fail('GENERAL_REVIEW_SCOPE','Historical scope differs');current=await loadScope(project);}catch(e){if(e.code!=='ENOENT')throw e;}
 }
 const checked=()=>{if(!current)fail('TASK_SCOPE_NOT_PREPARED','Use voder_task_prepare with intended source paths, then read returned SHA256 values.');return current;};
 const workspace=new HostTaskWorkspace({
  requireVisualDecision:()=>creationIntent&&!approval.readOnly,
  async prepare(paths,checks=current?.project.checks??[]){
   approval.requested=true; // An attempted write is never presented as a completed read-only answer.
   if(approval.readOnly)fail('TASK_READ_ONLY','The user requested read-only analysis');
   if(preparing)fail('TASK_SCOPE_BUSY','Wait for the in-flight scope preparation');
   if(scopeFrozen)fail('TASK_SCOPE_FROZEN','Verification/submission has begun; keep this file scope');
   if(!Array.isArray(paths)||!paths.length)fail('TASK_SCOPE_PATHS','Provide the intended relative source paths');
   const combined=[...new Set([...(current?.project.targets??[]),...paths])].sort();
   if(current&&JSON.stringify(combined)===JSON.stringify(current.project.targets)&&JSON.stringify(checks)===JSON.stringify(current.project.checks))return {status:'READY',files:await snapshot(cwd,combined),scope:combined};
   preparing=true;
   try{
    trace.write('prepare_plan_start',{file_count:combined.length});
    const plan=await planTask({agentRoot,storageRoot:authorization.storageRoot,root:cwd,scope:hostScope(combined,checks)});
    trace.write('prepare_plan_end');
    if(plan.root!==cwd)fail('TASK_REPOSITORY_BOUNDARY','These paths belong to a nested independent repository. Select that repository as the task folder; no new preimages were written.');
    trace.write('approval_request');
    const approved=await approval.request(activeSessionId,{root:plan.root,files:plan.scope.files,checks:plan.scope.checks,preimages:plan.before});
    trace.write('approval_decision',{approved:approved===true});
    if(!approved)fail('WRITE_SCOPE_DECLINED','The user declined or cancelled this exact write scope; no source preimages were created');
    if(approval.closed)fail('CANCELLED','Task cancelled before workspace preparation');
    if(authorization.approvalMode==='interactive')Object.assign(process.env,require('./native-env.cjs').nativeEnv(process.env));
    trace.write('scope_create_start');
    const project=await createTask(plan);trace.write('scope_create_end');trace.write('scope_load_start');current=await loadScope(project);trace.write('scope_load_end');
    await writeFile(join(dirname(gate),'active-scope.json'),JSON.stringify({profile:project.profile,recordPath:project.recordPath}),{mode:0o600});
    return {status:'READY',scope:project.targets,files:await snapshot(cwd,project.targets),checks:['immutable text/source checks inside independent sandbox'],visual_quality:'NOT_CERTIFIED',next_step:'Read relevant sources, construct candidate, render/inspect if requested, verify the exact final candidate; Host submits automatically after final-hash visual ACCEPT and authenticated PASS (nonvisual tasks retain explicit submission). Optional explicit argv checks may be configured with voder_task_prepare; they run in the independent sandbox only. No final planning JSON is needed.'};
   }catch(e){
    trace.error('prepare_error',e);
    if(['TASK_WORKSPACE_BLOCKED','GENERAL_WORKSPACE','EACCES','ENOENT','CHAT_NODE_RESOLVE_FAILED','CHAT_NODE_UNSAFE_DIRECTORY','CHAT_NODE_UNSAFE_EXECUTABLE'].includes(e.code)){blocker={code:e.code,reason:String(e.message).slice(0,600)};e.terminalTaskBlocker=true;}
    throw e;
   }finally{preparing=false;}
  },
  async render({path,content,preview_path,signal}){
   const label=typeof path==='string'&&path.length<=240?path:'<invalid SVG label>';renderCalls++;
   try{
    if(typeof path!=='string'||path.length>240||!/^[-A-Za-z0-9_][A-Za-z0-9._/-]*\.svg$/.test(path)||path.split('/').some(x=>!x||x==='.'||x==='..'))fail('SVG_LABEL','Use a safe relative .svg label. No file is read from this label.');
    // Rendering supplied text is read-only. Only a requested future file delivery needs write scope.
    if(preview_path!==undefined){
     if(approval.readOnly)fail('SVG_PREVIEW_REQUIRES_WRITES','Omit preview_path for read-only rendering. Saving a PNG requires a writable task.');
     const scope=checked().project.targets;
     if(!scope.includes(path))fail('SVG_SCOPE','Prepare this source path only when requesting saved preview delivery. Omit preview_path for read-only rendering.');
     if(!/^[A-Za-z0-9_-][A-Za-z0-9._/-]*\.png$/.test(preview_path)||preview_path.split('/').some(x=>!x||x==='.'||x==='..')||scope.includes(preview_path))fail('SVG_PREVIEW_PATH','Safe separate .png preview path required');
     await real(dirname(join(cwd,preview_path)));try{await lstat(join(cwd,preview_path));fail('SVG_PREVIEW_EXISTS','Preview already exists; choose a new path so unrelated binary files are preserved');}catch(e){if(e.code!=='ENOENT')throw e;}
    }
    const source_sha256=sha(content),rendered=await renderSvg(content,{agentRoot,signal});
    const record={path:label,status:'RENDERED',source_sha256,exposed:false,review:null};
    const row={record,path,source_sha256,preview_path,renderer:rendered.renderer,check:rendered.check,images:rendered.images,exposed:false,review:null};renders.set(source_sha256,row);while(renders.size>8)renders.delete(renders.keys().next().value);
    renderAttempts.set(label,record);
    const result={status:'RENDERED',path,source_sha256,renderer:row.renderer,technical:row.check,images:rendered.images.map((b,i)=>({scale:i?0.5:1,sha256:sha(b),bytes:b.length})),image_delivery:'NOT_EXPOSED_TO_MODEL',visual_quality:'NOT_YET_REVIEWED',project_files_written:false,preview_delivery:preview_path===undefined?'NOT_REQUESTED_READ_ONLY':'AFTER_VERIFIED_SOURCE_DELIVERY_ONLY'};
    return {result,images:row.images.map(b=>({type:'image',mimeType:'image/png',data:b.toString('base64')})),onExposed:()=>{result.image_delivery='FULL_AND_REDUCED_IMAGES_IN_TOOL_RESPONSE';},onRequested:()=>{row.exposed=true;record.exposed=true;}};
   }catch(error){renderAttempts.set(label,{path:label,status:'FAILED',code:/^[A-Z][A-Z0-9_]{0,95}$/.test(error.code??'')?error.code:'SVG_RENDER_FAILED'});throw error;}
  },
  async browser(args){
   const browserReviewNextStep=function(phaseApiAvailable){
    const required='NEXT ACTION: inspect the delivered real browser frames, then call voder_task_review_render with source_sha256 and one decision: ACCEPT, REPAIR_REQUIRED, or NEEDS_MORE_EVIDENCE. Report factual findings. ACCEPT only after actual visual examination; after ACCEPT verify the exact final candidate and let the authenticated Host/VGR commit it. REPAIR_REQUIRED requires source modification and a new inspection. NEEDS_MORE_EVIDENCE must identify a concrete novel check, not repeat the same browser call.';
    return required+(phaseApiAvailable?' Phase playback evidence is available; review its observed results.':' No application phase-controller API was available. Do not claim a phase sweep. Review the frames that WERE returned; use NEEDS_MORE_EVIDENCE only when a specific required interaction cannot yet be established.');
   }
   const task=validateBrowserInput(args),label=task.path,sourceDigest=sha(task.content),requestDigest=sha(JSON.stringify(task)),coverageDigest=sha(JSON.stringify({probe:task.probe??null,checks:task.checks,playback_seconds:task.playback_seconds})),attemptStarted=Date.now();let browserStarted=false,visualStarted=false;
   try{
    // Remote browser execution has a cost and requires actual file-scope approval in creation mode.
    // Explicit read-only inspection can use the same remote capability without workspace writes.
    if(creationIntent&&!current)fail('HTML_BROWSER_SCOPE_REQUIRED','Prepare the intended HTML source path before paid browser inspection');
    if(current&&!current.project.targets.includes(label))fail('HTML_BROWSER_SCOPE','HTML source is outside the approved file scope');
    if(task.preview_path!==undefined){
     if(approval.readOnly)fail('HTML_PREVIEW_REQUIRES_WRITES','Omit preview_path for read-only browser inspection');
     const scope=checked().project.targets;if(scope.includes(task.preview_path))fail('HTML_BROWSER_PATH','PNG is a derived preview, not a text candidate');
     await real(dirname(join(cwd,task.preview_path)));try{await lstat(join(cwd,task.preview_path));fail('HTML_PREVIEW_EXISTS','Preview exists; choose another path');}catch(e){if(e.code!=='ENOENT')throw e;}
    }
    // A repeated identical probe should recover the already captured real pixels,
    // not launch a fresh browser or demand a review of superseded source bytes.
    const cached=visualEvidenceReplay.request(requestDigest,sourceDigest,renders.get(sourceDigest));
    if(cached?.status==='REPLAY_FRAMES'){
     const row=cached.row,record=row.record;
     trace.write('browser_evidence_replayed',{request_sha256:requestDigest,source_sha256:sourceDigest,replays:cached.replays,new_browser_execution:false});
     const result={...row.check,status:'BROWSER_INSPECTED',source_sha256:sourceDigest,frames_reused:true,
      next_required_action:{tool:'voder_task_review_render',source_sha256:sourceDigest,required_fields:['source_sha256','findings','limitations','visual_decision']},
      next_step:'Real browser frames are reattached without a sandbox rerun. Judge significant visual/function defects only; record ACCEPT if requirements are met, REPAIR_REQUIRED for actual defects, or NEEDS_MORE_EVIDENCE for a specific untested requirement. Do not request this identical inspection again.',
      image_delivery:'NOT_EXPOSED_TO_MODEL',project_files_written:false};
     return {result,images:row.images.map(b=>({type:'image',mimeType:'image/png',data:b.toString('base64')})),
      onExposed:()=>{result.image_delivery='BROWSER_FRAMES_IN_TOOL_RESPONSE';},onRequested:()=>{record.exposed=true;row.exposed=true;}};
    }
    if(cached?.status==='REVIEW_REQUIRED'){
     const error=Object.assign(Error(`NO_NEW_BROWSER_EXECUTION. Exact browser frames have already been supplied for sha256 ${sourceDigest}; NEXT TOOL=voder_task_review_render with source_sha256=${sourceDigest}, findings, limitations and visual_decision=ACCEPT|REPAIR_REQUIRED|NEEDS_MORE_EVIDENCE. If this model cannot see images, select a configured vision-capable model; do not claim to have visually reviewed them.`),{code:'HTML_EVIDENCE_ALREADY_AVAILABLE'});
     if(cached.terminal)error.terminalTaskBlocker=true;
     throw error;
    }
    const decisionBlock=creationIntent&&!approval.readOnly?visualDecisionGate.before(label,sourceDigest,coverageDigest):null;
    if(decisionBlock){
     trace.write('visual_decision_required',{source_sha256:sourceDigest,code:decisionBlock.code,terminal:decisionBlock.terminal});
     const error=Object.assign(Error(decisionBlock.reason),{code:decisionBlock.code});
     if(decisionBlock.terminal)error.terminalTaskBlocker=true;
     throw error; // No remote browser created; retain the prior valid frames.
    }
    const replay=browserRepairGate.before(requestDigest);
    if(replay){
     trace.write('browser_replay_rejected',{request_sha256:requestDigest,source_sha256:sha(task.content),code:replay.code,terminal:replay.terminal});
     const error=Object.assign(Error(replay.reason),{code:replay.code});
     if(replay.terminal)error.terminalTaskBlocker=true;
     throw error;
    }
    if(!browserDriver&&authorization.approvalMode==='interactive')Object.assign(process.env,require('./native-env.cjs').nativeEnv(process.env));
    visualDecisionGate.starting(label,sourceDigest,coverageDigest);visualStarted=true;
    browserRepairGate.start(requestDigest);browserStarted=true;renderCalls++;browserExecutions++;
    trace.write('browser_attempt_started',{request_sha256:requestDigest,source_sha256:sha(task.content),attempt:browserExecutions});
    const inspected=await inspectHtml(task,{agentRoot,signal:args.signal,driver:browserDriver,onStage:status=>{for(const fn of approvalListeners.get(activeSessionId)??[])try{fn({sessionId:activeSessionId,type:'task.browser_stage',status});}catch{}}});
    if(browserAttemptHistory.length>=64)browserAttemptHistory.shift();browserAttemptHistory.push({attempt:browserExecutions,path:label,request_sha256:requestDigest,status:'BROWSER_INSPECTED',duration_ms:Date.now()-attemptStarted});
    const record={path:label,status:'RENDERED',kind:'HTML_BROWSER',probe:structuredClone(task.probe??null),source_sha256:inspected.result.source_sha256,exposed:false,review:null};
    const row={record,path:label,source_sha256:record.source_sha256,preview_path:task.preview_path,renderer:'Chromium remote browser',images:inspected.images,check:inspected.result,exposed:false,review:null};
    renders.set(row.source_sha256,row);while(renders.size>8)renders.delete(renders.keys().next().value);renderAttempts.set(label,record);
    const result={...inspected.result,status:'BROWSER_INSPECTED',next_step:browserReviewNextStep(inspected.result.phase_api_available===true),next_required_action:{tool:'voder_task_review_render',source_sha256:row.source_sha256,required_fields:['source_sha256','findings','limitations','visual_decision']},quality_decision_scope:'Only material visual defects and requested functional/animation behavior; optional phase API, cosmetic minutiae and wording length are NOT acceptance gates.',project_files_written:false,image_delivery:'NOT_EXPOSED_TO_MODEL',preview_delivery:task.preview_path?'AFTER_VERIFIED_SOURCE_DELIVERY_ONLY':'NOT_REQUESTED_READ_ONLY'};
    browserRepairGate.finish(requestDigest,true);browserStarted=false;
    visualEvidenceReplay.remember(requestDigest,sourceDigest,row);
    if(creationIntent&&!approval.readOnly)visualDecisionGate.inspected(label,row.source_sha256,coverageDigest);
    trace.write('browser_attempt_finished',{request_sha256:requestDigest,status:'BROWSER_INSPECTED'});
    return {result,images:row.images.map(b=>({type:'image',mimeType:'image/png',data:b.toString('base64')})),onExposed:()=>{result.image_delivery='BROWSER_FRAMES_IN_TOOL_RESPONSE';},onRequested:()=>{record.exposed=true;row.exposed=true;}};
   }catch(e){if(visualStarted)visualDecisionGate.failed(label,sourceDigest,coverageDigest);if(browserStarted){browserRepairGate.finish(requestDigest,false);trace.write('browser_attempt_finished',{request_sha256:requestDigest,status:'FAILED',code:e.code??'VODER_TOOL_FAILED'});}if(['HTML_BROWSER_REPAIR_REQUIRED','HTML_BROWSER_RETRY_UNPRODUCTIVE','HTML_BROWSER_ALREADY_RUNNING','HTML_VISUAL_DECISION_REQUIRED','HTML_VISUAL_REPAIR_NOT_APPLIED','HTML_VISUAL_ACCEPTED_NO_REINSPECT','HTML_BROWSER_NO_NEW_COVERAGE','HTML_BROWSER_SAME_SOURCE_BUSY','HTML_BROWSER_REVIEW_PENDING','HTML_EVIDENCE_ALREADY_AVAILABLE'].includes(e.code))throw e;if(e.code!=='HTML_BROWSER_BUDGET')if(browserAttemptHistory.length>=64)browserAttemptHistory.shift();browserAttemptHistory.push({attempt:browserExecutions,path:label,request_sha256:requestDigest,status:'FAILED',code:e.code??'VODER_TOOL_FAILED',reason:String(e.message??'Unknown browser failure').slice(0,1200),duration_ms:Date.now()-attemptStarted});if(['HTML_BROWSER_CONFIGURATION_ERROR','HTML_BROWSER_TIMEOUT','HTML_BROWSER_AUTH_REQUIRED','HTML_BROWSER_SETUP_FAILED','HTML_BROWSER_CLEANUP_UNCONFIRMED','MODULE_NOT_FOUND','ERR_MODULE_NOT_FOUND'].includes(e.code)){e.terminalTaskBlocker=true;blocker={code:e.code,reason:String(e.message).slice(0,600)};}renderAttempts.set(label,{path:label,status:'FAILED',kind:'HTML_BROWSER',code:e.code??'HTML_BROWSER_FAILED'});throw e;}
  },
  async review({source_sha256,findings,limitations,visual_decision}){
   const row=renders.get(source_sha256);
   if(!row||!row.exposed)fail('VISUAL_INPUT_UNAVAILABLE','No actual browser/SVG images for the requested source_sha256 were delivered to an image-capable model. Use the exact hash from the latest BROWSER_INSPECTED response, and ensure the configured vision model receives its images; do not repeat the paid browser inspection or invent visual ACCEPT.');
   if(typeof findings!=='string'||!findings.trim()||findings.length>6000||typeof limitations!=='string'||limitations.length>2000)fail('VISUAL_REVIEW_FORMAT','Bounded factual observations required');
   if(creationIntent&&!approval.readOnly&&row.record?.kind==='HTML_BROWSER'&&row.review?.visual_decision){
    const replay=visualDecisionGate.duplicateReview(row.path,source_sha256);
    const error=Object.assign(Error('Visual decision already recorded for these exact frames. ACCEPT: verify and commit the candidate; REPAIR_REQUIRED: change the HTML and re-inspect. Repeating the same review is not new evidence.'),{code:'VISUAL_REVIEW_ALREADY_DECIDED'});
    if(replay.terminal)error.terminalTaskBlocker=true;
    throw error;
   }
   if(creationIntent&&!approval.readOnly&&!['ACCEPT','REPAIR_REQUIRED','NEEDS_MORE_EVIDENCE'].includes(visual_decision))fail('VISUAL_REVIEW_DECISION','Review must choose ACCEPT, REPAIR_REQUIRED, or NEEDS_MORE_EVIDENCE based on actual images');
   if(creationIntent&&!approval.readOnly&&row.record?.kind!=='HTML_BROWSER'&&visual_decision==='NEEDS_MORE_EVIDENCE')fail('VISUAL_EVIDENCE_MODE','Additional browser evidence is only available for HTML inspection; SVG reviews must ACCEPT or REPAIR_REQUIRED');
   if(creationIntent&&!approval.readOnly&&row.record?.kind==='HTML_BROWSER'&&visual_decision==='NEEDS_MORE_EVIDENCE'&&!visualDecisionGate.reviewed(row.path,source_sha256,visual_decision,limitations))fail('VISUAL_EVIDENCE_PLAN_REQUIRED','State a concrete unmet visual/interaction requirement in limitations. Do not repeat the same unmet-evidence rationale after another inspection.');
   row.review={findings,limitations,visual_decision,authority:'MODEL_VISUAL_OBSERVATION_NOT_INDEPENDENT_QUALITY_CERTIFICATE'};row.record.review=row.review;
   if(creationIntent&&!approval.readOnly&&row.record?.kind==='HTML_BROWSER'&&visual_decision!=='NEEDS_MORE_EVIDENCE')visualDecisionGate.reviewed(row.path,source_sha256,visual_decision,limitations);
   return {status:'MODEL_REVIEW_RECORDED',source_sha256,renderer:row.renderer,...row.review,...(visual_decision==='REPAIR_REQUIRED'?{next_step:'Repair the identified defects in actual HTML/SVG source, inspect changed bytes, then verify the revised final candidate.'}:visual_decision==='ACCEPT'?{next_step:'Host will submit only after the exact candidate is independently PASSed and every final visual file is accepted.'}:visual_decision==='NEEDS_MORE_EVIDENCE'?{next_step:'Perform a genuinely new probe, check or playback observation to resolve the specific gap in limitations, then review the new evidence. Unchanged HTML is permitted only for novel inspection coverage; no Host commit is unlocked.'}:{})};
  },
  async commitReadiness(candidate){const c=checked();return evaluateVisualCommitReadiness(candidate,c.project.targets,renderAttempts,renders);},
  async hash(content){if(typeof content!=='string')fail('TASK_TEXT_LIMIT','Bounded text required');return {sha256:sha(content),bytes:Buffer.byteLength(content),written:false};},
  async assess({nativeCommit,delivery}){
   const rows=[],renderRows=[...renderAttempts.values()].map(x=>({path:x.path,status:x.status,...(x.kind?{kind:x.kind,probe:x.probe??null}:{}),...(x.code?{code:x.code}:{source_sha256:x.source_sha256,model_input:x.exposed?'SUBMITTED_BY_ADAPTER':'NOT_CONFIRMED',visual_review:x.review?'MODEL_REVIEW_RECORDED':'NOT_RECORDED'})})),failed=renderRows.some(x=>x.status==='FAILED');
   const rendering={requested:renderCalls>0,calls:renderCalls,status:!renderCalls?'NOT_REQUESTED':failed?(renderRows.some(x=>x.status==='RENDERED')?'PARTIAL':'FAILED'):'RENDERED',items:renderRows};
   if(nativeCommit?.status==='COMMITTED'&&delivery?.status==='DELIVERED'&&current){
    for(const path of current.project.targets){const data=(await regularBytes(join(cwd,path))).data,digest=sha(data),render=renders.get(digest),expected=delivery.files?.find(x=>x.relative_path===path)?.after_sha256??delivery.after_sha256;if(digest!==expected)fail('TASK_ARTIFACT_CHANGED','Source changed after verified delivery');rows.push({path,sha256:digest,...(/\.(?:svg|html)$/.test(path)?{rendered:!!render,visual_review:render?.review?'MODEL_REVIEW_RECORDED':'NOT_PERFORMED'}:{})});}
   }
   return {schema:'voder.task-acceptance.v1',...(blocker?{blocker}:{}),execution:approval.summary(),browser_executions:browserExecutions,browser_attempt_history:browserAttemptHistory.slice(-64),rendering,delivery:delivery?.status??'NOT_DELIVERED',configured_checks:nativeCommit?.status==='COMMITTED'?'PASSED':'NOT_PASSED',task_correctness:'NOT_INDEPENDENTLY_CERTIFIED',artifacts:[...rows,...artifacts],derived_preview:finalDelivery?.derived_preview??'NOT_DELIVERED',limitations:finalDelivery?.derived_preview_error??null};
  }
 });
 const candidateVerifier={getBudget:()=>({limit:null,used:0,remaining:null}),async verifyCandidate(candidate,options){
  const c=checked();await validateTask(c.project.recordPath,authorization.storageRoot);
  // Do not waste a sandbox check when the exact rendered candidate still lacks
  // its required model visual decision. No host-generated ACCEPT is permitted.
  if(creationIntent&&!approval.readOnly){
   const pending=visualReviewBeforeVerification(candidate,renderAttempts,renders);
   if(pending)fail(pending.code,`A browser image was exposed for ${pending.path} at sha256 ${pending.source_sha256}, but visual decision is ${pending.decision}. Call voder_task_review_render for this exact hash after examining the frames; do not repeat verification. A later ACCEPT does NOT bypass independent VGR.`);
  }
  // Scope, exact bytes and validator identity are checked again by HostCandidateVerifier and Rust.
  try{return await c.verifier.verifyCandidate(candidate,options);}
  catch(e){
   if(e?.code==='VERIFIER_SCOPE'){
    const error=Object.assign(Error(verifierScopeGuidance(candidate,c.project.targets)),{code:'VERIFIER_SCOPE',cause:e});
    throw error;
   }
   throw e;
  }
 }};
 const native={executeAuthorized:(input,options)=>{scopeFrozen=true;return checked().transport.executeAuthorized(input,options);},get targetPath(){return current?.cfg.targetPath;},get targetPaths(){return current?.cfg.targetPaths;}};
 workspace.policy=GENERAL_TASK_POLICY;
 const {HostToolRegistry}=await load('packages/voder-tools/src/index.mjs');
 const dialogueRegistry=require('./dialogue-model-tool.cjs').registerDialogueModelTool(new HostToolRegistry(),{projectRoot:authorization.memoryRoot??cwd,storageRoot:authorization.storageRoot});
 const sdk=new VoderSDK({cwd,native,candidateVerifier,toolRegistry:dialogueRegistry,taskWorkspace:workspace,commitMode:'evidence',searchOptions:{allowWeb:true,webMaxQueries:3},...(modelHost?{modelHost}:{}),modelProfiles:process.env.VODER_MODEL_PROFILES_JSON?JSON.parse(process.env.VODER_MODEL_PROFILES_JSON):[]});
 sdk.commitVerifier={confirmCommit:(receipt,input)=>checked().commitVerifier.confirmCommit(receipt,input)};
 sdk.verifiedDelivery={get attemptStore(){return current?.cfg.attemptStore;},get policyFile(){return current?.cfg.policyFile;},async deliver(args){
  const c=checked();finalDelivery=await c.delivery.deliver(args);
  if(finalDelivery.status!=='DELIVERED')return finalDelivery;
  // PNG is a host-rendered derivative, never an arbitrary model binary or a native receipt.
  for(const path of c.project.targets.filter(x=>/\.(?:svg|html)$/.test(x))){
   const data=(await regularBytes(join(cwd,path))).data,expected=(args.input.patches??[args.input]).find(x=>x.relative_path===path);if(!expected||sha(data)!==sha(expected.new_content)){finalDelivery={...finalDelivery,derived_preview:'NOT_DELIVERED',derived_preview_error:'SOURCE_CHANGED_AFTER_COMMIT'};continue;}const row=renders.get(sha(data));if(!row?.preview_path||row.path!==path)continue;
   const output=join(cwd,row.preview_path),temp=join(dirname(output),'.voder-preview-'+randomUUID());
   try{
    if(args.signal?.aborted)fail('CANCELLED','Preview delivery cancelled');
    await real(dirname(output));await writeFile(temp,row.images[0],{flag:'wx',mode:0o600});
    if(sha(await readFile(join(cwd,path)))!==row.source_sha256)fail('PREVIEW_SOURCE_CHANGED','SVG changed after verified source delivery');
    await link(temp,output); // Atomic no-clobber: never overwrite an existing user file.
    artifacts.push({path:row.preview_path,sha256:sha(row.images[0]),derived_from:path,source_sha256:row.source_sha256});finalDelivery={...finalDelivery,derived_preview:path.endsWith('.html')?'DELIVERED_FROM_EXACT_FINAL_HTML':'DELIVERED_FROM_EXACT_FINAL_SVG'};
   }catch(e){finalDelivery={...finalDelivery,derived_preview:'NOT_DELIVERED',derived_preview_error:String(e.code||'PREVIEW_DELIVERY_FAILED')};}
   finally{await unlink(temp).catch(()=>{});}
  }
  return finalDelivery;
 }};
 const journal=new VoderJournal({dir:sessionDir,workspace:cwd});await journal.initialize();
 const control=new VoderControlPlane({sdk,journal});
 const submit=control.submitTask.bind(control);control.submitTask=async(id,task)=>{creationIntent=!approval.readOnly&&writeRequest(task?.goal);return submit(id,task);};
 const create=control.createSession.bind(control);let sessionCreated=restored;
 control.createSession=async options=>{if(sessionCreated)fail('GENERAL_SESSION_CONSUMED','Each ordinary task has its own launch authorization and isolated task state');sessionCreated=true;await writeFile(reservationPath,JSON.stringify({schema:'voder.general-session.reservation.v1',automatic_replay:false}),{flag:'wx',mode:0o600});const created=await create({...options,autoContinue:false});activeSessionId=created.sessionId;return created;};
 const subscribe=control.subscribe.bind(control);
 control.subscribe=(id,callback,options)=>{const off=subscribe(id,callback,options),set=approvalListeners.get(id)??new Set();set.add(callback);approvalListeners.set(id,set);const p=approval.pending;if(p?.sessionId===id)callback({type:'task.workspace_approval_requested',sessionId:id,requestId:p.requestId,scopeDigest:p.scopeDigest,root:p.scope.root,files:p.scope.files,checks:p.scope.checks});return()=>{off();set.delete(callback);if(!set.size)approvalListeners.delete(id);};};
 control.authorizeWorkspace=input=>{trace.write('approval_response_received',{approved:input?.approved===true});return approval.decide(input);};
 const cancel=control.cancel.bind(control);control.cancel=async id=>{if(id===activeSessionId)approval.close();return cancel(id);};
 const wait=control.wait.bind(control);control.wait=async id=>{const result=await wait(id);if(id===activeSessionId)approval.close();return result;};
 const shutdown=control.shutdown.bind(control);control.shutdown=async()=>{approval.close();return shutdown();};
 const dispose=control.dispose.bind(control);control.dispose=()=>{approval.close();approvalListeners.clear();dispose();};
 // Keep sandbox/native repair diagnostics in the live task; no automatic replay on infrastructure failure.
 return control;
}
export const GENERAL_TASK_POLICY=`Operational checkpoint for HTML creation: after collecting sufficient browser frames for the FINAL HTML hash, make one genuine voder_task_review_render decision before final verification. Revised source may be re-inspected without a pointless review of obsolete frames. Only significant visual/function defects and explicit user requirements justify repairs or additional probes; optional phase APIs and cosmetic minutiae are not independent quality gates. A failed browser call is not visual evidence; correct that failure instead of inventing a review. After ACCEPT use exact-candidate verification; Rust VGR still decides commit. For HTML/web animation tasks, use voder_task_inspect_html with the exact complete HTML after preparing source scope. It executes JavaScript in a separate network-blocked remote browser, not on the host. Supply optional probe function paths for your own application debug API; the host does not assume a scene-specific global. Use returned real phase images and numeric page results; for file-creation tasks, for the final HTML evidence call voder_task_review_render with exact source_sha256 and a deliberate visual_decision: ACCEPT, REPAIR_REQUIRED, or NEEDS_MORE_EVIDENCE. NEEDS_MORE_EVIDENCE should identify the materially missing requirement; novel inspection coverage is allowed without pre-filing a review of incomplete frames, but still does not unlock commit. If a visual defect exists, select REPAIR_REQUIRED, change the HTML, then re-inspect. Browser phase checks do not mean ACCEPT. Optional page-side JavaScript checks are evidence from an untrusted app, not verifier authority. Browser inspection never replaces independent candidate verification or Rust commit. Static SVG rendering alone cannot verify an HTML animation, controls or playback. Request a derived PNG via preview_path on HTML inspection, not as a text source. If the host reports a terminal preparation blocker, the task stops without another paid model request; the host reports the exact reason. Execute the user's original task in the selected workspace. Planning is advisory; no separate JSON planning reply is required. Use repository tools to inspect only relevant files. For creation or modification, call voder_task_prepare with intended source paths, read returned preimage hashes, then verify the exact final candidate; Host submits automatically after exact-hash visual ACCEPT and authenticated PASS (nonvisual work retains explicit commit). All existing native authorization and independent Rust checks remain required. Preserve unrelated files. Source candidates are UTF-8 text (1..16 paths, no fixed whole-file byte ceiling; use paginated reads and exact hash-pinned edits); do not disguise binary data as text. For SVG assessment, call voder_task_render_svg with the exact supplied SVG text and a safe relative .svg label, omitting preview_path. This is read-only in-memory rendering, needs no voder_task_prepare or write approval, and does not read a workspace file at that label. Use it even when the SVG exists only in an attachment. Only set preview_path when the user requests saved PNG delivery and the source write scope is prepared. Do not ask for write approval just to inspect rendered images. A renderer failure must be stated explicitly; source-coordinate estimates do not replace visual inspection. Optional SVG PNG previews are produced only by voder_task_render_svg and delivered as derivatives of the exact verified source. Optional exact argv checks can be configured with voder_task_prepare and run ONLY inside the independent sandbox. Arbitrary binary formats and arbitrary HOST command execution are not available in this host; state any actual blocker and preserve valid editable source. A PASS is evidence for configured checks only, not proof that the user task is finished. Complete requested rendering and full/reduced visual inspection BEFORE final commit if available; repair significant defects and rerender. Record visual findings with voder_task_review_render only after actually seeing the images. A text-only model cannot claim visual inspection. Hash exact final bytes with voder_task_content_hash when writing verification records. For HTML/SVG, record voder_task_review_render with visual_decision ACCEPT only after genuinely reviewing final-hash frames; use REPAIR_REQUIRED for defects. Host deterministically submits the exact verified PASS candidate only when every final visual artifact is ACCEPT. For non-visual tasks, call voder_commit_verified_candidate explicitly with the final checked digest. Do not claim a visual quality certificate or skip independent Rust VGR. Do not substitute a webpage or animation for requested SVG or change the subject. Respect any user prohibition on writing/executing. Read-only questions need no workspace write approval. The first real file-scope preparation may wait for an explicit user decision; never infer approval from model text. For image assessment, separate observed details from uncertain interpretations. A component not clearly visible is not proven absent: say it is unclear. Inspect an appropriate crop only if an actual crop tool is available; otherwise state that limitation. Do not invent pixel distances, timings or objective quality scores. Label any requested subjective score as subjective. Do not infer the generating model or architecture from image appearance; SVG/code generation is not evidence of diffusion generation. Use provided source/reference evidence when available; never invent having inspected a crop, file, renderer or test. Report artifact paths, checks actually performed, and limitations concisely.`;
async function main(){
 const [sub,...args]=process.argv.slice(2);if(sub!=='rpc')throw Error('GENERAL_RPC_ONLY');const flags={};
 for(let i=0;i<args.length;i+=2){if(!['--workspace','--session-dir','--gateway-config','--agent-root'].includes(args[i])||typeof args[i+1]!=='string'||Object.hasOwn(flags,args[i]))throw Error('GENERAL_INVALID_FLAGS');flags[args[i]]=args[i+1];}
 process.umask(0o077);const control=await makeGeneralControl({cwd:flags['--workspace'],sessionDir:flags['--session-dir'],gate:flags['--gateway-config'],agentRoot:flags['--agent-root']});
 const {serveRpc}=await import(pathToFileURL(join(flags['--agent-root'],'packages/voder-cli/bin/voder-cli.mjs')).href);await serveRpc(control);
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url))main().catch(e=>{process.stderr.write('VODER_GENERAL_START_BLOCKED='+String(e.code||'CONTROL_START_FAILED')+'\n');process.exitCode=1;});
