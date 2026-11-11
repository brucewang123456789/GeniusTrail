import {createTrace} from './private-trace.mjs';
import {isHostTaskTool,taskToolImages,hasTaskToolImages,taskToolRequestAck} from './task-workspace.mjs';
import referenceContext from './reference-context.cjs';
import {installOutputRecovery} from './output-recovery.mjs';
import {contextTokens,capacityReserves,compactionSettings} from './context-capacity.mjs';
import {guardedStream} from './request-policy.mjs';
import {resolvePiImports} from './pi-imports.mjs';
import {ACTION_SCHEMA} from './patch-schema.mjs';
import {createHash} from 'node:crypto';
import {isHostApprovedReadTool} from '../../voder-tools/src/index.mjs';
import { resolve } from 'node:path';
import { VoderError } from './contracts.mjs';
import { bindInstalledModel } from './model-plugin-runtime.mjs';

/** Preserve upstream model loop/session/compaction; Voder owns the entire tool boundary. */
const ALLOWED = new Set(['voder_authorized_action','voder_read_file','voder_list_files','voder_search_code','voder_verify_candidate','voder_commit_verified_candidate','voder_memory_propose']);
const ACTION_PARAMS = {type:'object',properties:{action:ACTION_SCHEMA},required:['action'],additionalProperties:false};
const toolShape = tool => tool.name==='voder_authorized_action' ? ACTION_PARAMS : tool.parameters;
function inspectDeclaration(tools){
  if(!Array.isArray(tools)||tools.length<1||tools.length>ALLOWED.size+14+tools.filter(isHostTaskTool).length ||
     !tools.some(x=>x?.name==='voder_authorized_action') ||
     new Set(tools.map(x=>x?.name)).size!==tools.length ||
     tools.some(x=>!(ALLOWED.has(x?.name)||isHostApprovedReadTool(x)||isHostTaskTool(x))||typeof x.execute!=='function' || typeof x.description!=='string'||
       (x.name!=='voder_authorized_action' && (!x.parameters || x.parameters.type!=='object'))))
    throw new VoderError('INVALID_TOOL_SET','Only explicitly VGR-governed Voder tools may be registered');
}
// Only *repeated deterministic rejects* are bounded; normal model reasoning,
// novel evidence, browser runs and independent VGR verification remain unrestricted.
export function createVerifierRejectionGuard(){
 let scope=null,scopeErrors=0,commitErrors=new Map();const repeated=new Map();
 const digest=payload=>createHash('sha256').update(JSON.stringify(payload)).digest('hex');
 return {
  scopeReady(paths){
   if(!Array.isArray(paths))return;
   const next=digest(paths);
   if(next!==scope){scope=next;scopeErrors=0;repeated.clear();commitErrors.clear();}
  },
  verified(status){if(status==='PASS'){scopeErrors=0;repeated.clear();commitErrors.clear();}},
  denied({tool,code,payload,scopeMismatchClass}){
   if(tool==='voder_verify_candidate'&&code==='VERIFIER_SCOPE'){
    const key=digest({tool,code,payload,scope});const same=(repeated.get(key)??0)+1;
    repeated.set(key,same);scopeErrors++;
    if(scopeMismatchClass==='HOST_POLICY_MISMATCH')return 'HOST_VERIFIER_SCOPE_CONFIGURATION';
    if(same>=2||scopeErrors>=3)return 'VERIFIER_SCOPE_REPEATED';
   }
   if(['voder_authorized_action','voder_commit_verified_candidate'].includes(tool)&&code==='VERIFY_BEFORE_COMMIT'){
    const key=digest({tool,code,payload,scope});const same=(commitErrors.get(key)??0)+1;
    commitErrors.set(key,same);
    if(same>=2)return 'PREMATURE_COMMIT_REPEATED';
   }
   return null;
  }
 };
}
export class PiModelHost {
 constructor({upstream,modelProfiles=[],maxModelRequests=64}={}){if(!Number.isSafeInteger(maxModelRequests)||maxModelRequests<1||maxModelRequests>64)throw new VoderError('MODEL_REQUEST_BUDGET','Invalid host request budget');this.upstream=upstream;this.modelProfiles=modelProfiles;this.maxModelRequests=maxModelRequests;}
 async createSession({cwd,systemPrompt,tools,toolControl}){
   inspectDeclaration(tools);
   const requestLimit=tools.some(t=>t.name==='voder_task_inspect_html')?Infinity:this.maxModelRequests;
   if(toolControl!==undefined&&typeof toolControl!=='function')throw new VoderError('INVALID_ORCHESTRATION_POLICY','Host tool control must be callable');
   const api=this.upstream??await import('@earendil-works/pi-coding-agent');
   const {createAgentSession,DefaultResourceLoader,SessionManager,getAgentDir,defineTool}=api;
   if([createAgentSession,DefaultResourceLoader,getAgentDir,defineTool].some(x=>typeof x!=='function') || typeof SessionManager?.inMemory!=='function')
     throw new VoderError('INCOMPATIBLE_UPSTREAM','Pinned upstream SDK contract changed');
   const base=resolve(cwd);const trace=createTrace(base);trace.write('model_host_open');
   const loader=new DefaultResourceLoader({cwd:base,agentDir:getAgentDir(),noExtensions:true,noSkills:true,
     noPromptTemplates:true,noContextFiles:true,systemPromptOverride:()=>systemPrompt,appendSystemPromptOverride:()=>[]});
   await loader.reload();
   const injectedUpstream = this.upstream !== undefined;
   // Voder already applies provider-specific image capacity and records its hashes.
   // Keep all other effective SDK settings, but disable a second implicit resize.
   let preparedImageSettings;
   if(api.SettingsManager?.create && api.SettingsManager?.inMemory){
     const inherited=api.SettingsManager.create(base,getAgentDir()).getSettings();
     preparedImageSettings=api.SettingsManager.inMemory({...inherited,images:{...inherited.images,autoResize:false}});
     if(preparedImageSettings.getImageAutoResize()!==false)throw new VoderError('INCOMPATIBLE_UPSTREAM','Cannot preserve prepared image bytes');
   }else if(!injectedUpstream)throw new VoderError('INCOMPATIBLE_UPSTREAM','Image settings contract unavailable');

   const modelBinding=injectedUpstream?null:await bindInstalledModel(api,process.env,this.modelProfiles);
   if(preparedImageSettings&&modelBinding){
    const inherited=api.SettingsManager.create(base,getAgentDir()).getSettings();
    preparedImageSettings=api.SettingsManager.inMemory(compactionSettings(inherited,modelBinding.scopedModels.map(x=>x.model)));
   }
   let nativeCommitted=false,evidenceEpoch=0,terminalFailure=null,finalizationProgress=null,stalledFinalizationRequests=0;const verifierGuard=createVerifierRejectionGuard();
   const control=()=>toolControl?.();
   const beforeRequest=()=>{
     if(nativeCommitted)return;
     if(terminalFailure)throw terminalFailure;
     // A bounded Pi summarizer call consumes request capacity, not an agent tool turn.
     if(session.isCompacting)return;
     const ledger=control();
     if(ledger?.phase!=='FINALIZING')return;
     // Count MODEL turns without a new verification/action, never concurrent read errors.
     // Unrelated reads cannot reset this finite recovery allowance.
     const progress=JSON.stringify([ledger.verification_calls??0,ledger.attempted_native_actions??0,ledger.task_evidence_revisions??0]);
     stalledFinalizationRequests=progress===finalizationProgress?stalledFinalizationRequests+1:0;
     finalizationProgress=progress;
     if(stalledFinalizationRequests>=2){terminalFailure=new VoderError('FINALIZATION_STALLED','Finalization made no verification or native submission progress across two model turns; no automatic replay');throw terminalFailure;}
   };
   const pendingImageAcks=[];
   const evidenceSeen=new Set();
   // Keep the transcript intact. Only superseded host browser frames are omitted from
   // provider requests; user attachments and distinct source paths remain byte-exact.
   const browserCalls=new Map(),latestBrowserCall=new Map();
   const allImages=context=>context.messages.flatMap(m=>Array.isArray(m.content)?m.content.filter(x=>x.type==='image'):[]);
   const assertImages=(found,expected)=>{if(expected&&(found.length!==expected.length||found.some((x,i)=>x.data!==expected[i].data||x.mimeType!==expected[i].mimeType)))throw new VoderError('REFERENCE_IMAGE_CHANGED','Image was omitted or transformed by the model adapter; no request sent');};
   const browserContext=context=>({...context,messages:context.messages.map(m=>{
    const registered=m.role==='toolResult'?browserCalls.get(m.toolCallId):null;
    // Retain complementary probes of the SAME source bytes. Identical probes
    // and obsolete source versions are superseded to bound image context.
    const latestId=latestBrowserCall.get(registered?.path),latest=browserCalls.get(latestId);
    const complementary=latest?.sourceSha256&&latest.sourceSha256===registered?.sourceSha256&&latest.coverageSha256!==registered.coverageSha256;
    if(!registered||!latest||latestId===m.toolCallId||complementary||!Array.isArray(m.content))return m;
    const found=m.content.filter(x=>x.type==='image');assertImages(found,registered.images);
    return {...m,content:[...m.content.filter(x=>x.type!=='image'),{type:'text',text:'Superseded browser frames omitted from this request; use the latest inspection for this source path. Original transcript and source hash are retained.'}]};
   })});
   const customTools=tools.map(t=>defineTool({
     name:t.name,label:t.name,description:t.description,parameters:toolShape(t),
     execute:async(_callId,args,signal)=>{
       let payload; // Share the original tool argument with the guarded error path.
       try{
         if(terminalFailure&&!nativeCommitted)throw terminalFailure;
         payload=t.name==='voder_authorized_action'?args.action:args;
         const result=await t.execute(payload,signal);
         if((['voder_authorized_action','voder_commit_verified_candidate'].includes(t.name)&&result?.status==='COMMITTED')||(t.name==='voder_verify_candidate'&&result?.native_commit?.confirmed===true))nativeCommitted=true;
         if(t.name==='voder_task_prepare'&&result?.status==='READY')verifierGuard.scopeReady(result.scope);
         if(t.name==='voder_verify_candidate')verifierGuard.verified(result?.status);
         if(['DENIED','BLOCKED'].includes(result?.status)){
          const code=result.error_code??result.code;
          const stop=verifierGuard.denied({tool:t.name,code,payload,scopeMismatchClass:result.scopeMismatchClass});
          if(stop&&!nativeCommitted)terminalFailure=new VoderError(stop,'Deterministic verification rejection repeated without a changed authorized scope; no automatic model replay');
         }
         // Count only host-completed, novel tool observations; model prose and rejected partial calls cannot reset recovery.
         if(t.name!=='voder_memory_propose'&&result!==null&&result!==undefined&&!['DENIED','BLOCKED'].includes(result?.status)){const key=createHash('sha256').update(JSON.stringify({tool:t.name,args,result})).digest('hex');if(!evidenceSeen.has(key)){evidenceSeen.add(key);evidenceEpoch++;}}
         // Native receipt remains untrusted until the independent Rust verifier confirms it.
         const safe=t.name==='voder_authorized_action'?{status:result.status,action_id:result.action_id}:result;
         const ledger=control();
         if(isHostTaskTool(t)&&hasTaskToolImages(result)&&!session.model?.input?.includes('image')){
          // Try every ALREADY-APPROVED vision model deterministically (dedicated
          // provider first). No new credentials, providers or unapproved models.
          const candidates=approved.filter(m=>m.input?.includes('image')).sort((a,b)=>Number(b.provider==='voder-secondary-vision')-Number(a.provider==='voder-secondary-vision'));
          const previous=session.model,before=JSON.stringify(session.messages??[]);let selected=false;
          for(const target of candidates){
           if(typeof session.setModel!=='function')break;
           try{
            if(modelBinding&&!(await modelBinding.modelRuntime.checkAuth(target.provider)))continue;
            await session.setModel(target,{persist:false});const thinking=thinkingFor(target);if(thinking!==undefined)session.setThinkingLevel?.(thinking,{persist:false});policyCheck();
            if(session.model?.provider!==target.provider||session.model?.id!==target.id||JSON.stringify(session.messages??[])!==before)throw new VoderError('MODEL_SWITCH_INVARIANT','Vision selection changed the task transcript');
            modelEpoch++;selected=true;emitHost({type:'vision_selection',provider:target.provider,model:target.id,status:'CONFIGURED_TOOL_VISION_SELECTED'});break;
           }catch(error){
            trace.write('vision_candidate_unavailable',{provider:target.provider,model:target.id,code:String(error?.code??'MODEL_SELECT_FAILED')});
            try{await session.setModel(previous,{persist:false});policyCheck();if(session.model?.provider!==previous?.provider||session.model?.id!==previous?.id||JSON.stringify(session.messages??[])!==before)throw Error('RESTORE_MISMATCH');}
            catch{quarantined=true;throw new VoderError('MODEL_SWITCH_RECOVERY_FAILED','Cannot restore model after failed vision selection');}
           }
          }
          if(!selected){
           // Do not fake image delivery or terminate a valid work session here.
           // The browser's factual evidence remains available for source repair;
           // only an actual vision-capable model may submit visual ACCEPT.
           result.image_delivery='VISION_MODEL_NOT_CONFIGURED_OR_UNAVAILABLE';
           result.visual_review_status='AWAITING_REAL_IMAGE_DELIVERY';
           result.next_step='The browser succeeded, but these PNG frames were not delivered to a vision-capable model. Configure an approved vision model, or use the objective browser results for code repair; do NOT claim visual ACCEPT or launch identical browser checks. Real screenshot evidence is necessary before final visual approval.';
           emitHost({type:'vision_delivery',status:'VISION_MODEL_NOT_CONFIGURED_OR_UNAVAILABLE',tool:t.name});
          }
         }
         const renderedImages=isHostTaskTool(t)?taskToolImages(result,{supported:session.model?.input?.includes('image')}):[];
         if(renderedImages.length&&t.name==='voder_task_inspect_html'){browserCalls.set(_callId,{path:args.path,sourceSha256:result?.source_sha256,coverageSha256:createHash('sha256').update(JSON.stringify([args.probe??null,args.checks??[],args.playback_seconds??0])).digest('hex'),images:renderedImages.map(x=>({data:x.data,mimeType:x.mimeType}))});latestBrowserCall.set(args.path,_callId);}
         if(renderedImages.length){expectedImages=[...(expectedImages??[]),...renderedImages.map(x=>({data:x.data,mimeType:x.mimeType}))];const ack=taskToolRequestAck(result);if(ack)pendingImageAcks.push({ack,callId:_callId,...(t.name==='voder_task_inspect_html'?{browserPath:args.path}:{})});}
         return {content:[{type:'text',text:JSON.stringify({voder_tool_result:safe,...(ledger?{tool_control:ledger}:{})})},...renderedImages],details:{status:result?.status??'READ_OK'}};
       }catch(e){
         trace.error('tool_error',e,{tool:t.name,evidence_epoch:evidenceEpoch});
       const code=typeof e?.code==='string'?e.code:'VODER_TOOL_FAILED';
         const ledger=control();
         if(!nativeCommitted&&isHostTaskTool(t)&&e?.terminalTaskBlocker===true)terminalFailure=e;
         if(!nativeCommitted&&['voder_authorized_action','voder_commit_verified_candidate'].includes(t.name)&&['NATIVE_REJECTED','NATIVE_OUTCOME_UNCERTAIN','OUTCOME_UNKNOWN'].includes(code)){e.terminalTaskBlocker=true;terminalFailure=e;}
         if(!nativeCommitted&&(code==='TOOL_BUDGET_EXHAUSTED'||(['VERIFY_BUDGET_EXHAUSTED','VERIFIER_BUDGET'].includes(code)&&ledger?.positive_sandbox_checks===0)))terminalFailure=new VoderError(code,'Authoritative capacity exhausted without a PASS; no further model request or task replay');
         const stop=verifierGuard.denied({tool:t.name,code,payload,scopeMismatchClass:e?.scopeMismatchClass});
         if(stop&&!nativeCommitted)terminalFailure=new VoderError(stop,'Deterministic verification rejection repeated without a changed authorized scope; no automatic model replay');
         return {content:[{type:'text',text:JSON.stringify({error:code,reason:isHostApprovedReadTool(t)?({SOURCE_NOT_DISCOVERED:'Search first, then fetch an exact returned URL.',WEB_DISABLED:'Host web retrieval is disabled.',WEB_FORBIDDEN_BY_TASK:'The user forbids web retrieval.',WEB_DOMAIN_DENIED:'Host domain policy denies this source.',WEB_QUERY_BUDGET:'Discovery or total request limit reached; try fetch if its reserved request remains.',TOOL_CIRCUIT_OPEN:'Adapter failure threshold reached; inspect tool receipts.',WEB_DNS_FAILED:'DNS resolution failed before connection; inspect DNS trace.',DNS_PUBLIC_ADDRESS_REQUIRED:'DNS did not yield an exclusively approved public address set; no connection permitted.',WEB_SOURCE_NOT_FOUND:'The server returned 404/410; select another URL. This does not prove that the requested fact or event does not exist.',WEB_ACCESS_DENIED:'The remote server denied access; select another source.',WEB_PROVIDER_CIRCUIT_OPEN:'This operation is cooling down; another source or fetch may still work.',WEB_PROVIDER_FAILED:'The provider was invoked but failed; this is not a local evidence prerequisite.'}[code]??'Read-only adapter refused or failed; inspect the error code and receipts. Do not infer a network outage from a policy refusal.'):String(e?.message??'Tool failed').slice(0,480),...(ledger?{tool_control:ledger}:{})})}],
           details:{status:'DENIED',error_code:code}};
       }
     }
   }));
   // Exact allowlist is critical: Pi v1.1.0 codemode/deferred/MCP tools can be callable without being active.
   const {session}=await createAgentSession({cwd:base,sessionManager:SessionManager.inMemory(base),resourceLoader:loader,
     noTools:'all',tools:tools.map(t=>t.name),customTools,
     ...(preparedImageSettings?{settingsManager:preparedImageSettings}:{}),
     ...(modelBinding?{model:modelBinding.model,modelRuntime:modelBinding.modelRuntime,scopedModels:modelBinding.scopedModels,thinkingLevel:modelBinding.policies[0]?.thinkingLevel??'off'}:{})});
   const unsubscribeTrace=session.subscribe?.(event=>trace.observe(event));
   try{const retry=preparedImageSettings?.getRetrySettings?.();trace.write('engine_configuration',{config_retry_enabled:retry?.enabled,config_retry_count:retry?.maxRetries,config_retry_delay_ms:retry?.baseDelayMs,thinking_level:session.thinkingLevel??'unknown'});}catch{}
   const expected=new Set(tools.map(t=>t.name));
   const inspect=()=>{
     if(typeof session.getActiveToolNames!=='function'||typeof session.getCallableToolNames!=='function'||typeof session.getAllTools!=='function')
       throw new VoderError('INCOMPATIBLE_UPSTREAM','Tool introspection unavailable');
     const active=session.getActiveToolNames();const callable=session.getCallableToolNames();const registered=session.getAllTools();
     const exact=arr=>Array.isArray(arr)&&arr.length===expected.size&&new Set(arr).size===arr.length&&arr.every(n=>expected.has(n));
     if(!exact(active)||!exact(callable)||!Array.isArray(registered)||registered.length!==expected.size ||
       registered.some(t=>!expected.has(t.name)||t.exposure!=='direct') || new Set(registered.map(t=>t.name)).size!==registered.length)
       throw new VoderError('UNSAFE_TOOLS','Upstream active/callable/registered tools exceed VGR allowlist');
     return Object.freeze({active:[...active],callable:[...callable],registered:registered.map(t=>t.name)});
   };
   try{inspect();}catch(e){session.dispose();throw e;}
   let operation=null,disposed=false,compactCount=0,switchCount=0,modelEpoch=0,quarantined=false;const switches=[];
   const baseline=typeof session.systemPrompt==='string'?session.systemPrompt:null;
   const approved=modelBinding?.scopedModels.map(x=>x.model)??(injectedUpstream?this.modelProfiles.map(x=>({...x,provider:x.provider??session.model?.provider??'test'})):[]);
   if(session.model&&!approved.some(x=>x.provider===session.model.provider&&x.id===session.model.id))approved.unshift(session.model);
   const policies=modelBinding?.policies??[];const thinkingFor=m=>policies.find(p=>p.provider===m?.provider&&p.id===m?.id)?.thinkingLevel;
   let expectedImages=null;
   const requests=[];let streamGuard,recovery,pendingFailure=null;const listeners=new Set();
   const emitHost=e=>{trace.host(e);if(e.type==='output_recovery'&&e.code==='MODEL_TRUNCATION_STALLED')pendingFailure=e.code;for(const fn of listeners){try{fn(e);}catch{}}};
   if(!injectedUpstream||session.agent?.streamFunction){
    if(typeof session.agent?.streamFunction!=='function')throw new VoderError('INCOMPATIBLE_UPSTREAM','Pi request stream hook unavailable');
    let normalize=api.normalizeContext;
    if(!injectedUpstream){const {aiEntry}=await resolvePiImports(import.meta.url);const ai=await import(aiEntry);normalize=ai.normalizeContext;}
    if(typeof normalize!=='function')throw new VoderError('INCOMPATIBLE_UPSTREAM','Branded transcript normalizer unavailable');
    const rawStream=session.agent.streamFunction.bind(session.agent);
    let requestImages=null;const providerGuard=guardedStream({maxRequests:requestLimit,raw:(model,context,options)=>{if(!model?.input?.includes('image')&&context.messages.some(m=>Array.isArray(m.content)&&m.content.some(x=>x.type==='image')))throw new VoderError('REFERENCE_IMAGE_UNSUPPORTED','Selected model cannot retain image context');if(requestImages){const found=allImages(context);assertImages(found,requestImages);emitHost({type:'image_request',provider:model.provider,model:model.id,image_count:found.length,images_sha256:createHash('sha256').update(JSON.stringify(found.map(x=>({mimeType:x.mimeType,sha256:createHash('sha256').update(Buffer.from(x.data,'base64')).digest('hex')})))).digest('hex'),status:'ADAPTER_VALIDATED_NOT_PROVIDER_ACK'});}trace.write('provider_dispatch',{provider:model.provider,model:model.id,effective_max_tokens:options?.maxTokens,thinking_level:typeof options?.reasoning==='string'?options.reasoning:session.thinkingLevel??'unknown'});const stream=trace.transport(()=>rawStream(model,context,options));for(const entry of pendingImageAcks.splice(0))if(!entry.browserPath||latestBrowserCall.get(entry.browserPath)===entry.callId)entry.ack();return stream;},normalizeContext:normalize,systemPrompt,inspect,allowedModels:approved,currentModel:()=>session.model,thinkingFor,onPressure:r=>{if(!nativeCommitted)toolControl?.(r.code);emitHost({type:'context_pressure',...r});},onFailure:(code,detail)=>{pendingFailure=code;emitHost({type:'model_failure',code,...detail});},onAudit:r=>{if(pendingFailure==='CONTEXT_OVERFLOW')pendingFailure=null;requests.push(r);if(!nativeCommitted&&Number.isFinite(requestLimit)&&r.sequence>requestLimit-Math.min(4,requestLimit-1))toolControl?.('MODEL_FINALIZATION_RESERVE');emitHost({type:'request_start',...r,request_sequence:r.sequence});}});
    streamGuard=(model,context,options)=>{
     beforeRequest();
     try{assertImages(allImages(context),expectedImages);}catch(error){terminalFailure=error;throw error;}
     const projected=browserContext(context);requestImages=expectedImages?allImages(projected):null;
     try{return providerGuard(model,projected,options);}finally{requestImages=null;}
    };
    streamGuard.beginModelStage=()=>providerGuard.beginModelStage();session.agent.streamFunction=streamGuard;
   }
   const policyCheck=()=>{inspect();if(streamGuard&&session.agent.streamFunction!==streamGuard)throw new VoderError('REQUEST_GUARD_CHANGED','Request policy guard changed');if(baseline!==null&&(session.systemPrompt!==baseline||!session.systemPrompt.includes(systemPrompt)))throw new VoderError('SYSTEM_POLICY_CHANGED','Trusted system rules changed during session maintenance');};
   if(!injectedUpstream||typeof session.agent?.steer==='function'){
    // Preserve Pi's turn-end lifecycle, then close a confirmed native commit before any next request.
    const previousFinishTurn=session.agent.finishTurn;
    session.agent.finishTurn=async function(turn,signal){
     const decision=await previousFinishTurn?.call(this,turn,signal);
     return nativeCommitted||terminalFailure?{action:'end'}:decision;
    };
    recovery=installOutputRecovery(session.agent,{check:policyCheck,onEvent:emitHost,mayContinue:()=>!nativeCommitted&&!terminalFailure,getProgress:()=>evidenceEpoch});
   }
   let compactionLedger;
   const compactionCheckpoint=()=>{const c=control();return c?{phase:c.phase,ledger:JSON.stringify(Object.fromEntries(Object.entries(c).filter(([key])=>!['phase','phase_reason','next_step'].includes(key))))}:null;};
   const unsubscribeCapacity=session.subscribe?.(event=>{
    if(event.type==='compaction_start'){compactionLedger=compactionCheckpoint();emitHost({type:'compaction_start',status:'STARTED',reason:event.reason??'upstream'});}
    if(event.type==='compaction_end'){const after=compactionCheckpoint();const unchanged=compactionLedger?.ledger===after?.ledger&&!(compactionLedger?.phase==='FINALIZING'&&after?.phase==='EXPLORING');compactionLedger=undefined;
     if(!unchanged&&!nativeCommitted)terminalFailure=new VoderError('SESSION_LEDGER_CHANGED','Compaction changed authoritative task evidence or budget; no further action permitted');
     emitHost({type:'compaction_end',status:event.error?'FAILED':event.aborted?'ABORTED':'COMPLETED',ledger_unchanged:unchanged});try{policyCheck();}catch(error){terminalFailure=error;}}
   });
   const reserve=kind=>{if(quarantined)throw new VoderError('SESSION_QUARANTINED','Model transition could not be restored; create a new session');if(disposed||operation||session.isStreaming||session.isCompacting)throw new VoderError('SESSION_BUSY','Session operation already active');policyCheck();recovery?.inspect();operation=kind;};
   const policyHash=()=>createHash('sha256').update(systemPrompt).digest('hex');
   const info=()=>({provider:session.model?.provider,id:session.model?.id,system_prompt_sha256:policyHash(),model_epoch:modelEpoch,pricing_status:'UNCONFIGURED_NO_DOLLAR_ESTIMATE'});
   return {
     referenceContextImages:true,
     async prompt(goal,options){reserve('prompt');pendingFailure=null;if(!injectedUpstream&&!modelBinding)throw new VoderError('MODEL_PLUGIN_REQUIRED','Install provider plugin and supply model ID and key');try{const images=options?.images;const retainedImages=(session.messages??[]).flatMap(m=>Array.isArray(m.content)?m.content.filter(x=>x?.type==='image'):[]);if(images?.length||retainedImages.length){if(!session.model?.input?.includes('image')){
 const candidates=approved.filter(m=>m.input?.includes('image'));
 const dedicated=candidates.filter(m=>m.provider==='voder-secondary-vision');
 const target=dedicated.length===1?dedicated[0]:candidates.length===1?candidates[0]:null;
 if(!target)throw new VoderError('REFERENCE_IMAGE_UNSUPPORTED','No unambiguous configured vision model. Run Voder: Configure Vision Model; document images were retained and no text-only fallback was sent');
 if(typeof session.setModel!=='function')throw new VoderError('REFERENCE_IMAGE_UNSUPPORTED','Runtime cannot select the configured vision model');
 if(modelBinding&&!(await modelBinding.modelRuntime.checkAuth(target.provider)))throw new VoderError('MODEL_AUTH_UNAVAILABLE','Vision model credentials unavailable');
 const before=JSON.stringify(session.messages??[]),previous=session.model;
 try{await session.setModel(target,{persist:false});const thinking=thinkingFor(target);if(thinking!==undefined)session.setThinkingLevel?.(thinking,{persist:false});policyCheck();if(session.model?.provider!==target.provider||session.model?.id!==target.id||JSON.stringify(session.messages??[])!==before)throw new VoderError('MODEL_SWITCH_INVARIANT','Vision selection changed the transcript');modelEpoch++;}
 catch(error){try{await session.setModel(previous,{persist:false});}catch{quarantined=true;}throw error;}
 emitHost({type:'vision_selection',provider:target.provider,model:target.id,status:'CONFIGURED_CAPABILITY_SELECTED'});
 }if(!streamGuard)throw new VoderError('REFERENCE_IMAGE_UNSUPPORTED','Image integrity guard unavailable');referenceContext.validateReferences([...retainedImages,...(images||[])].map((x,i)=>({id:'00000000-'+i,kind:'image',name:'image-'+i,source:'attachment',data:x.data,mimeType:x.mimeType,sha256:referenceContext.digest(Buffer.from(x.data,'base64'))})));expectedImages=[...retainedImages,...(images||[])].map(x=>({data:x.data,mimeType:x.mimeType}));}if(modelBinding?.limits)emitHost({type:'model_policy',limit_source:modelBinding.limits.source,capability_verified:modelBinding.limits.capability_verified,registered_max_tokens:session.model?.maxTokens,context_window:session.model?.contextWindow});await (images?.length?session.prompt(goal,{images}):session.prompt(goal));}finally{expectedImages=null;operation=null;policyCheck();}
       const messages=Array.isArray(session.messages)?session.messages:[];
       const last=[...messages].reverse().find(m=>m?.role==='assistant');
       if(terminalFailure&&!nativeCommitted)throw terminalFailure;
       if(pendingFailure)throw new VoderError(pendingFailure,'Host recovery stopped without new evidence; no task or native action was replayed');
       if(last?.stopReason==='error')throw new VoderError('MODEL_RESPONSE_ERROR','Upstream model session ended with an error; inspect private metadata diagnostics');
       if(last?.stopReason==='aborted')throw new VoderError('CANCELLED','Model response was aborted');
       const draft=!['stop','end_turn'].includes(last?.stopReason)?undefined:
         Array.isArray(last?.content)?last.content.filter(b=>b?.type==='text'&&typeof b.text==='string').map(b=>b.text).join('\n'):undefined;
       return {settled:true,...(draft&&Buffer.byteLength(draft,'utf8')<=131072?{draft}:{})};},
     async compact(options={}){
       if(!options||Object.keys(options).some(k=>k!=='instructions')||(options.instructions!==undefined&&(typeof options.instructions!=='string'||Buffer.byteLength(options.instructions)>4096)))throw new VoderError('INVALID_COMPACTION','Bounded operator instructions required');
       if(compactCount>=2)throw new VoderError('SESSION_MAINTENANCE_BUDGET','Compaction attempt budget exhausted');
       if((session.messages??[]).some(m=>Array.isArray(m.content)&&m.content.some(x=>x?.type==='image')))throw new VoderError('REFERENCE_IMAGE_COMPACTION_UNSUPPORTED','Compaction cannot discard source image coverage; start a new task with the original files');
       reserve('compact');compactCount++;try{if(typeof session.compact!=='function')throw new VoderError('INCOMPATIBLE_UPSTREAM','Compaction API absent');
        await session.compact(options.instructions);policyCheck();evidenceEpoch++;return {status:'COMPACTED',...info()};
       }catch(e){if(e instanceof VoderError)throw e;if(/Nothing to compact|Already compacted/.test(String(e?.message)))return {status:'NO_COMPACTION_NEEDED',...info()};throw new VoderError('COMPACTION_FAILED','Upstream compaction failed; inspect private provider diagnostics before another attempt');}finally{operation=null;}
     },
     async switchModel(request){
       if(!request||Array.isArray(request)||Object.keys(request).some(k=>!['provider','id'].includes(k))||typeof request.provider!=='string'||typeof request.id!=='string')throw new VoderError('INVALID_MODEL_SELECTION','Only an explicit registered model identity is accepted');
       const model=approved.find(m=>m.provider===request.provider&&m.id===request.id);
       if(!model)throw new VoderError('MODEL_NOT_ALLOWED','Model is not in this session approved registry');
       reserve('switch');let previous,previousThinking,beforeContext;
       try{
        previous=session.model;previousThinking=session.thinkingLevel;
        if(previous?.provider===model.provider&&previous?.id===model.id)return {status:'MODEL_UNCHANGED',...info()};
        const estimated=contextTokens(model,{messages:[{role:'system',content:systemPrompt},...(session.messages??[])],tools:customTools.map(({name,description,parameters})=>({name,description,parameters}))});
        const reserve=capacityReserves(model);
        if(!Number.isSafeInteger(model.contextWindow)||!Number.isSafeInteger(model.maxTokens)||estimated.tokens+64+reserve.safety+reserve.finalization>model.contextWindow)throw new VoderError('CONTEXT_OVERFLOW','Target token estimate leaves insufficient reserved capacity; compact explicitly or choose a larger model');
        if(modelBinding&&!(await modelBinding.modelRuntime.checkAuth(model.provider)))throw new VoderError('MODEL_AUTH_UNAVAILABLE','Target provider credential is unavailable');
        if(typeof session.setModel!=='function')throw new VoderError('INCOMPATIBLE_UPSTREAM','Model-switch API absent');
        beforeContext=JSON.stringify(session.messages??[]);switchCount++;
        try{
         await session.setModel(model,{persist:false});
         const thinking=thinkingFor(model);if(thinking!==undefined){if(typeof session.setThinkingLevel!=='function')throw new VoderError('INCOMPATIBLE_UPSTREAM','Target thinking configuration unavailable');session.setThinkingLevel(thinking,{persist:false});if(session.thinkingLevel!==thinking)throw new VoderError('MODEL_THINKING_MISMATCH','Target thinking level was not applied');}
         policyCheck();if(session.model?.id!==model.id||session.model?.provider!==model.provider||JSON.stringify(session.messages??[])!==beforeContext)throw new VoderError('MODEL_SWITCH_INVARIANT','Switch must preserve the exact conversation and selected identity');
        }catch(error){
         try{await session.setModel(previous,{persist:false});if(previousThinking!==undefined)session.setThinkingLevel(previousThinking,{persist:false});policyCheck();if(session.model?.id!==previous?.id||session.model?.provider!==previous?.provider||JSON.stringify(session.messages??[])!==beforeContext)throw Error('RESTORE_MISMATCH');}
         catch{quarantined=true;switches.push({status:'QUARANTINED',from:{provider:previous?.provider,id:previous?.id},to:{provider:model.provider,id:model.id}});throw new VoderError('MODEL_SWITCH_RECOVERY_FAILED','No further model requests are permitted in this session');}
         switches.push({status:'ROLLED_BACK',from:{provider:previous?.provider,id:previous?.id},to:{provider:model.provider,id:model.id}});throw new VoderError('MODEL_SWITCH_ROLLED_BACK','Target selection failed; previous model and context restored');
        }
        modelEpoch++;evidenceEpoch++;terminalFailure=null;pendingFailure=null;finalizationProgress=null;stalledFinalizationRequests=0;streamGuard?.beginModelStage?.();const receipt={status:'MODEL_SELECTED',previous:{provider:previous?.provider,id:previous?.id},...info(),context_sha256:createHash('sha256').update(beforeContext).digest('hex')};switches.push(receipt);return receipt;
       }finally{operation=null;}
     },
     async listModels(){if(quarantined)throw new VoderError('SESSION_QUARANTINED','Model transition requires a new session');policyCheck();return {current:info(),maintenance_budget:{compaction_attempts:compactCount,max_compaction_attempts:2,model_switches:switchCount,max_model_switches:null},models:approved.map(m=>({provider:m.provider,id:m.id,contextWindow:m.contextWindow,maxTokens:m.maxTokens}))};},
     requestAudit(){return requests.map(r=>({...r}));},switchAudit(){return structuredClone(switches);},
     inspectTools:inspect,async abort(){await session.abort();},dispose(){unsubscribeTrace?.();unsubscribeCapacity?.();disposed=true;session.abortCompaction?.();session.dispose();},
     subscribe(fn){listeners.add(fn);const unsubscribe=session.subscribe(fn);return()=>{listeners.delete(fn);unsubscribe?.();};}
   };
 }
}
