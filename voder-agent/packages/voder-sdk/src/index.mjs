import {BatchWorkspaceDelivery} from './batch-delivery.mjs';
import {VerifiedWorkspaceDelivery} from './verified-delivery.mjs';
import { NativeVgrCommitVerifier } from '../../voder-native-bridge/src/commit-verifier.mjs';
import {TrustedVgrGatewayTransport,GatewayCommitVerifier} from '../../voder-native-bridge/src/trusted-gateway.mjs';
import { randomUUID } from 'node:crypto';
import { AgentKernel } from '../../voder-agent-core/src/kernel.mjs';
import { PiModelHost } from '../../voder-agent-core/src/pi-host.mjs';
import { VgrBroker } from '../../voder-agent-core/src/broker.mjs';
import { VoderError } from '../../voder-agent-core/src/contracts.mjs';
import {HostToolRegistry} from '../../voder-tools/src/index.mjs';
import {InstalledSkills} from '../../voder-skills/src/index.mjs';
import {EvidenceSearchService} from '../../voder-search/src/index.mjs';
import { createVoderContextProvider, inferExplicitRequiredFiles } from '../../voder-context/src/index.mjs';
import {MemoryProposalQueue,exportMemoryFile,importMemoryFile,renderMemoryContext} from '../../voder-memory/src/index.mjs';
/** Public Voder API; no SDK API can synthesize a VGR final-commit receipt. */
export class VoderSDK {
 constructor({cwd, native, nativeBinary, nativeStateDir, nativeGateway, modelHost,modelProfiles=[], contextProvider, contextOptions, contextSignals, candidateVerifier, orchestrationOptions,taskWorkspace,commitMode, allowUnverifiedDraft=false, toolRegistry,toolEconomyOptions,skillStore,explicitSkills=[],diagnosticOptions,searchOptions,onEvent}={}){
   if(typeof cwd!=='string'||!cwd)throw new VoderError('INVALID_WORKSPACE','Missing workspace');
   if(contextProvider!==undefined && typeof contextProvider!=='function')throw new VoderError('INVALID_CONTEXT_PROVIDER','Trusted contextProvider must be a function');
   if(contextOptions!==undefined && (contextOptions===null||typeof contextOptions!=='object'||Array.isArray(contextOptions)))
     throw new VoderError('INVALID_CONTEXT_POLICY','contextOptions must be a plain configuration object');
   if(onEvent!==undefined && typeof onEvent!=='function')throw new VoderError('INVALID_EVENT_SINK','Event sink must be a function');
   this.listeners=new Map();this.onEvent=event=>{try{onEvent?.(event);}catch{}for(const listener of this.listeners.get(event.session_id)??[])try{listener(event);}catch{}};
   this.cwd=cwd;this.native=nativeGateway ? new TrustedVgrGatewayTransport(nativeGateway) : native;this.modelHost=modelHost??new PiModelHost({modelProfiles});
   this.contextProvider=contextProvider??createVoderContextProvider(contextOptions);
   if(toolRegistry!==undefined && !(toolRegistry instanceof HostToolRegistry))throw new VoderError('INVALID_TOOL_REGISTRY','Tool registry must be explicitly host-installed');
   if(skillStore!==undefined && !(skillStore instanceof InstalledSkills))throw new VoderError('INVALID_SKILL_STORE','Skill store must be administrator-selected');
   this.toolRegistry=toolRegistry;this.toolEconomyOptions=toolEconomyOptions;this.skillStore=skillStore;this.explicitSkills=explicitSkills;
   if(searchOptions!==undefined && (searchOptions===null||typeof searchOptions!=='object'||Array.isArray(searchOptions)))
     throw new VoderError('INVALID_SEARCH_POLICY','searchOptions must be a host configuration object');
   this.searchService=searchOptions===undefined?undefined:new EvidenceSearchService({cwd,...searchOptions});
   this.diagnosticOptions=diagnosticOptions;
   this.taskWorkspace=taskWorkspace;this.commitMode=commitMode;
   this.contextSignals=contextSignals;this.candidateVerifier=candidateVerifier;this.allowUnverifiedDraft=allowUnverifiedDraft;this.orchestrationOptions=orchestrationOptions;this.sessions=new Map();this.memoryQueue=new MemoryProposalQueue();
   // The commit verifier is host-owned; the model cannot choose this binary,
   // daemon state directory, or a claimed final-commit digest.
   this.verifiedDelivery=nativeGateway ? new (nativeGateway.targetPaths?BatchWorkspaceDelivery:VerifiedWorkspaceDelivery)({cwd,attemptStore:nativeGateway.attemptStore,policyFile:nativeGateway.policyFile}) : null;
   this.commitVerifier = nativeGateway
     ? new GatewayCommitVerifier({daemonBinary:nativeGateway.daemonBinary,attemptStore:nativeGateway.attemptStore})
     : nativeBinary && nativeStateDir ? new NativeVgrCommitVerifier({binary:nativeBinary,stateDir:nativeStateDir}) : undefined;
 }
 async preflight(){return this.native instanceof TrustedVgrGatewayTransport ? this.native.preflight() : null;}
 createSession({id=randomUUID(),memoryFiles=[],enableMemoryProposals=false}={}){
   if(this.sessions.has(id))throw new VoderError('DUPLICATE_SESSION','Duplicate session id');
   const broker=new VgrBroker({native:this.native,sessionId:id,workspaceRoot:this.cwd,commitVerifier:this.commitVerifier,delivery:this.verifiedDelivery});
   const kernel=new AgentKernel({modelHost:this.modelHost,broker,cwd:this.cwd,sessionId:id,contextProvider:this.contextProvider,contextSignals:this.contextSignals,
     taskWorkspace:this.taskWorkspace,commitMode:this.commitMode,candidateVerifier:this.candidateVerifier,orchestrationOptions:this.orchestrationOptions,allowUnverifiedDraft:this.allowUnverifiedDraft,
     toolRegistry:this.toolRegistry,toolEconomyOptions:this.toolEconomyOptions,skillStore:this.skillStore,explicitSkills:this.explicitSkills,
     diagnosticOptions:this.diagnosticOptions,searchService:this.searchService,memoryFiles,memoryProposals:enableMemoryProposals,memoryQueue:this.memoryQueue,onEvent:this.onEvent});
   this.sessions.set(id,kernel);return {sessionId:id};
 }
 _get(id){const k=this.sessions.get(id);if(!k)throw new VoderError('NOT_FOUND','Unknown session');return k;}
 async submitTask(id,task){
   // Conservative preflight: explicit attachment and @path markers only, never guessed paths.
   // Existing trusted requiredFiles remains authoritative. The kernel blocks before model calls.
   const inferred=inferExplicitRequiredFiles(task?.goal);
   const declared=task?.requiredFiles??[];
   if(!Array.isArray(declared))throw new VoderError('INVALID_REQUIRED_FILES','requiredFiles must be an array');
   const session=this._get(id);
   return session.submit({...task,requiredFiles:[...new Set([...declared,...inferred])]});
 }
 /** Per-session, ephemeral model proposals. Each entry is user-review data, not persisted. */
 reconcileBatchDelivery(id,actionId){if(typeof this.verifiedDelivery?.reconcile!=='function')throw new VoderError('BATCH_RECOVERY_UNAVAILABLE','Batch delivery profile required');return this.verifiedDelivery.reconcile({actionId,sessionId:id,commitVerifier:this.commitVerifier});}
 continueTask(id){return this._get(id).continueTask();}
 compactSession(id,options){return this._get(id).maintainSession('compact',options);}
 switchModel(id,options){return this._get(id).maintainSession('switchModel',options);}
 listModels(id){return this._get(id).maintainSession('listModels');}
 getMemoryProposals(id){this._get(id);return this.memoryQueue.list(id);}
 /** The user must confirm exactly which proposed IDs become a portable file. */
 async approveMemoryProposal(id,proposalId,{selectedIds,outputPath,profile,userConfirmed=false}={}){
   this._get(id);
   return this.memoryQueue.approve({sessionId:id,proposalId,selectedIds,outputPath,profile,userConfirmed});
 }
 /** Manual import/export can run without an active model, remote calls or native VGR authority. */
 async exportMemory(options){return exportMemoryFile(options);}
 async inspectMemoryFile(path){return importMemoryFile(path);}
 async previewMemoryContext(paths){
   if(!Array.isArray(paths)||paths.length>8)throw new VoderError('INVALID_MEMORY_SELECTION','At most 8 paths');
   const bundles=[];for(const path of paths)bundles.push(await importMemoryFile(path));
   return renderMemoryContext(bundles);
 }
 /** Trusted host hook: durable dispatch intent, never exposes native authority to a model. */
 setNativeDispatchHooks(id,{before,after}={}){
   const broker=this._get(id).broker;
   if(broker.beforeNativeDispatch||broker.afterNativeDispatch)throw new VoderError('HOOKS_ALREADY_SET','Dispatch hooks already registered');
   if(typeof before!=='function'||typeof after!=='function')throw new VoderError('INVALID_DISPATCH_HOOKS','Both hooks required');
   broker.beforeNativeDispatch=before;broker.afterNativeDispatch=after;
 }
 getEvents(id,{after=0}={}){if(!Number.isInteger(after)||after<0)throw new VoderError('INVALID_CURSOR','Invalid cursor');return this._get(id).events.filter(e=>e.sequence>after);}
 subscribeEvents(id,listener,{after=0}={}){
   this._get(id);if(typeof listener!=='function'||!Number.isSafeInteger(after)||after<0)throw new VoderError('INVALID_SUBSCRIBER','Invalid event subscriber');
   const set=this.listeners.get(id)??new Set();this.listeners.set(id,set);
   for(const event of this.getEvents(id,{after}))try{listener(event);}catch{}
   set.add(listener);return ()=>{set.delete(listener);if(!set.size)this.listeners.delete(id);};
 }
 getStatus(id){return this._get(id).state;}
 async cancel(id){return this._get(id).cancel();}
 dispose(){this.listeners.clear();for(const [id,k] of this.sessions){k.close();this.memoryQueue.clear(id);}this.sessions.clear();}
}

export {VoderControlPlane, VoderJournal} from './control-plane.mjs';
export {NativeSdkContinuation} from './native-continuation.mjs';
