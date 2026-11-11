#!/usr/bin/env node
import referenceContext from '../../voder-agent-core/src/reference-context.cjs';
/** Editor-neutral CLI + JSON-lines RPC adapter. Side effects remain owned by the existing Rust VGR transport. */
import {createInterface} from 'node:readline';
import {readFile,lstat,realpath} from 'node:fs/promises';
import {resolve,join,dirname} from 'node:path';
import {homedir} from 'node:os';
import {fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';
import {HostToolRegistry} from '../../voder-tools/src/index.mjs';
const require=createRequire(import.meta.url);
import {VoderSDK,VoderControlPlane,VoderJournal,NativeSdkContinuation} from '../../voder-sdk/src/index.mjs';
import {HostCandidateVerifier} from '../../voder-sdk/src/candidate-verifier.mjs';
import {VoderError} from '../../voder-agent-core/src/contracts.mjs';
import {readNativeSessionDiff} from '../../voder-sdk/src/native-diff.mjs';
import {GatewayCommitVerifier} from '../../voder-native-bridge/src/trusted-gateway.mjs';

const here=dirname(fileURLToPath(import.meta.url));
const defaultJournal=join(homedir(),'.local','state','voder','sessions-v1');
const paramLimit=65536;
function writeHumanResult(output,result,sessionId,journalDir){
 output.write(`VODER_SESSION=${sessionId}\nVODER_STATUS=${result.status}\nVODER_JOURNAL=${journalDir}\n`);
 if(result.status==='DRAFT_UNVERIFIED'&&result.verified===false&&typeof result.draft==='string'){
  output.write('\n[UNVERIFIED MODEL DRAFT — NOT VGR COMMITTED]\n');
  output.write(result.draft);output.write(result.draft.endsWith('\n')?'':'\n');
 }else output.write('VODER_MODEL_ANSWER=NOT_AVAILABLE\n');
}

const fail=(code,reason)=>{throw new VoderError(code,reason)};
function parse(args){
 const flags={};const pos=[];
 for(let i=0;i<args.length;i++){
  const arg=args[i];if(arg.startsWith('--')){
   const p=arg.slice(2).split('=',2),key=p[0];if(!/^[a-z][a-z-]*$/.test(key))fail('INVALID_OPTION','Invalid option');
   if(Object.hasOwn(flags,key)&&key!=='memory')fail('DUPLICATE_OPTION',key);
   const val=p.length===2?p[1]:(args[i+1]&&!args[i+1].startsWith('--')?args[++i]:true);
   if(key==='memory'){flags.memory??=[];flags.memory.push(val);}else flags[key]=val;
  }else pos.push(arg);
 }
 return {flags,pos};
}
function allowed(flags,keys){for(const k of Object.keys(flags))if(!keys.includes(k))fail('UNSUPPORTED_OPTION',k);}
function str(v,name){if(typeof v!=='string'||!v.trim()||v.length>4096)fail('INVALID_OPTION',`${name} must have a value`);return v;}
function envPath(){return process.env.VODER_SESSION_DIR||defaultJournal;}
async function workspace(path){const absolute=resolve(path||process.cwd());const real=await realpath(absolute);const st=await lstat(real);
 if(!st.isDirectory()||st.isSymbolicLink())fail('INVALID_WORKSPACE','Workspace must be a directory');return real;}
async function gatewayConfig(file){
 if(!file)return null;
 const abs=resolve(file);const st=await lstat(abs);
 if(!st.isFile()||st.isSymbolicLink()||st.size>8192||(st.mode&0o077))fail('INVALID_GATEWAY_CONFIG','Gateway config must be a private regular file (0600)');
 const v=JSON.parse(await readFile(abs,'utf8'));
 const keys=['binary','daemonBinary','policyFile','attemptStore','nodeBinary','modalCli','targetPath'];
 if(!v||typeof v!=='object'||Object.keys(v).some(k=>!keys.includes(k)&&k!=='targetPaths')||keys.some(k=>typeof v[k]!=='string'))fail('INVALID_GATEWAY_CONFIG','Explicit host-owned VGR gateway paths are required');
 return v;
}
function nativeDeny(){return {async executeAuthorized(){fail('NATIVE_VGR_NOT_CONFIGURED','Native VGR gateway not configured; no executable write permission');}};}
export async function makeControl({cwd,sessionDir,gate,draft=false,modelHost,web=false,dialogueStorageRoot}={}){
 const path=await workspace(cwd);const config=await gatewayConfig(gate);
 if(draft&&config)fail('DRAFT_GATEWAY_CONFLICT','Draft-only cannot be presented as verified VGR mode');
 if(!draft&&!config)fail('NATIVE_VGR_REQUIRED','Configure trusted VGR gateway or explicitly request --draft (unverified text only)');
 const candidateVerifier=config?new HostCandidateVerifier({cwd:path,nativeGateway:config}):undefined;
 let dialogueRegistry;
 if(dialogueStorageRoot!==undefined){
  if(typeof dialogueStorageRoot!=='string'||!dialogueStorageRoot.startsWith('/')||await realpath(dialogueStorageRoot)!==dialogueStorageRoot)fail('MEMORY_STORAGE_ROOT_INVALID','Dialogue storage root must be a canonical host directory');
  dialogueRegistry=require('../../voder-vscode/src/dialogue-model-tool.cjs').registerDialogueModelTool(new HostToolRegistry(),{projectRoot:path,storageRoot:dialogueStorageRoot});
 }
 const sdk=new VoderSDK({cwd:path,modelProfiles:process.env.VODER_MODEL_PROFILES_JSON?JSON.parse(process.env.VODER_MODEL_PROFILES_JSON):[],...(config?{nativeGateway:config}:{native:nativeDeny()}),
   allowUnverifiedDraft:draft,...(dialogueRegistry?{toolRegistry:dialogueRegistry}:{}),...(web?{searchOptions:{allowWeb:true,webMaxQueries:3}}:{}),...(candidateVerifier?{candidateVerifier}:{}),...(modelHost?{modelHost}:{})});
 if(config)await sdk.preflight();
 if(candidateVerifier)candidateVerifier.onInfrastructureError=id=>sdk.cancel(id);
 const journal=new VoderJournal({dir:sessionDir||envPath(),workspace:path});
 await journal.initialize();
 const nativeContinuation=config?new NativeSdkContinuation({gatewayBinary:config.binary,daemonBinary:config.daemonBinary,attemptStore:config.attemptStore}):undefined;
 return new VoderControlPlane({sdk,journal,nativeContinuation});
}
function send(stream,obj){stream.write(JSON.stringify(obj)+'\n');}
function paramsObject(p){if(p===undefined)return {};if(!p||typeof p!=='object'||Array.isArray(p))fail('INVALID_PARAMS','Expected params object');return p;}
/** Stable line-framed protocol for Node, Python, VS Code and future editors. */
export async function serveRpc(control,{input=process.stdin,output=process.stdout}={}){
 let shut=false;const active=new Set();const subscriptions=new Map();
 const route=async(method,p)=>{
  switch(method){
   case 'initialize':return {protocol:'voder.control.v1',interface:'jsonl-stdio',capabilities:{
     workspace_scope_approval:typeof control.authorizeWorkspace==='function',reference_context:'voder.references.v1',reference_frame_bytes:referenceContext.LIMITS.wireBytes,create:true,submit:true,events:true,stream:true,cancel_live:true,compaction:true,model_switch:true,same_task_continue:true,multi_file_transaction:true,memory_approval:true,metadata_export:true,
     history_read_only:true,native_resume:Boolean(control.nativeContinuation),native_recovery:'rust_reconciliation_only',session_scoped_diff:Boolean(control.sdk.verifiedDelivery&&control.sdk.commitVerifier),vgr_commit:'native_only'}};
   case 'session.reconcileBatchDelivery':return control.reconcileBatchDelivery(p.sessionId,p.actionId);
   case 'session.continue':return control.continueSession(p.sessionId);
   case 'session.compact':return control.compactSession(p.sessionId,p.options);
   case 'session.models':return control.listModels(p.sessionId);
   case 'session.switchModel':return control.switchModel(p.sessionId,p.model);
   case 'session.create':{
     if(p.autoContinue===true&&!control.nativeContinuation)fail('NATIVE_RECOVERY_REQUIRED','Recovery opt-in requires operator-configured native VGR');
     return control.createSession({memoryFiles:p.memoryFiles??[],enableMemoryProposals:p.enableMemoryProposals??false,autoContinue:p.autoContinue??false});
   }
   case 'session.list':return control.journal.list();
   case 'task.submit':if(p.task?.referenceContext!==undefined)referenceContext.validateReferences(p.task.referenceContext);return control.submitTask(p.sessionId,p.task);
   case 'task.authorizeWorkspace':if(typeof control.authorizeWorkspace!=='function')fail('CAPABILITY_UNAVAILABLE','Interactive workspace authorization is not enabled');return control.authorizeWorkspace(p);
   case 'task.wait':return control.wait(p.sessionId);
   case 'session.status':return control.status(p.sessionId);
   case 'session.events':return control.events(p.sessionId,{after:p.after??0});
   case 'session.subscribe':{
     if(subscriptions.has(p.sessionId))fail('ALREADY_SUBSCRIBED','Unsubscribe before subscribing again');
     const unsubscribe=control.subscribe(p.sessionId,e=>send(output,{jsonrpc:'2.0',method:'session.event',params:{sessionId:p.sessionId,event:e}}),{after:p.after??0});
     subscriptions.set(p.sessionId,unsubscribe);return {sessionId:p.sessionId,status:'SUBSCRIBED'};
   }
   case 'session.unsubscribe':{const f=subscriptions.get(p.sessionId);if(f){f();subscriptions.delete(p.sessionId);}return {sessionId:p.sessionId,status:'UNSUBSCRIBED'};}
   case 'session.cancel':return control.cancel(p.sessionId);
   case 'session.resume':return control.resume(p.sessionId); // Legacy read-only contract must never trigger tools.
   case 'session.recover':{
     if(!control.nativeContinuation)fail('NATIVE_RECOVERY_REQUIRED','A trusted Rust VGR gateway is required for guarded automatic recovery');
     const sessionId=p.sessionId;
     if(typeof sessionId!=='string'||!p.task||typeof p.task!=='object'||Array.isArray(p.task))fail('RECOVERY_TASK_REQUIRED','Exact host-resolved task and UUID are required');
     const results=await control.autoRecover({sessionId,taskResolver:async()=>p.task});
     return results[0];
   }
   case 'session.diff':return control.getDiff(p.sessionId);
   case 'session.export':return control.export(p.sessionId,p.outputPath);
   case 'memory.proposals':return control.memoryProposals(p.sessionId);
   case 'memory.approve':return control.approveMemory(p.sessionId,p.proposalId,{selectedIds:p.selectedIds,outputPath:p.outputPath,profile:p.profile,userConfirmed:p.userConfirmed});
   default:fail('METHOD_NOT_FOUND',`Unknown operation: ${method}`);
  }
 };
 const handle=async(line)=>{
  let req,id=null;
  try{
   if(Buffer.byteLength(line)>referenceContext.LIMITS.wireBytes)fail('FRAME_TOO_LARGE','Request frame exceeded absolute limit');
   req=JSON.parse(line);id=req?.id??null;
   if(Buffer.byteLength(line)>paramLimit&&(req?.method!=='task.submit'||!req?.params?.task?.referenceContext?.length))fail('FRAME_TOO_LARGE','Legacy request frame exceeded limit');
   if(req?.jsonrpc!=='2.0'||!['number','string'].includes(typeof req.id)||typeof req.method!=='string'||req.method.length>64)fail('INVALID_REQUEST','JSON-RPC id, method and version required');
   const result=await route(req.method,paramsObject(req.params));
   send(output,{jsonrpc:'2.0',id:req.id,result});
  }catch(e){send(output,{jsonrpc:'2.0',id,error:{code:e?.code??'INTERNAL_ERROR',message:String(e?.message??'request failed').slice(0,240)}});}
 };
 const rl=createInterface({input,crlfDelay:Infinity});
 for await(const line of rl){if(!line.trim())continue;if(shut)break;const task=handle(line).finally(()=>active.delete(task));active.add(task);}
 shut=true;await Promise.allSettled(active);for(const f of subscriptions.values())f();
 await control.shutdown();
}
function help(){return `Voder terminal/SDK control plane (v1)\n
voder run --goal TEXT [--workspace PATH] [--draft] [--web] [--gateway-config FILE] [--memory FILE ...] [--format json|text]\nvoder rpc [--workspace PATH] [--draft] [--web] [--gateway-config FILE]   # JSON-RPC 2.0 over stdio lines\nvoder status --session UUID [--workspace PATH]\nvoder events --session UUID [--after N] [--workspace PATH]\nvoder resume --session UUID [--workspace PATH]   # inspect history, never automatically re-run tools\nvoder diff --session UUID --gateway-config FILE [--workspace PATH]  # rust-verified read-only diff\nvoder export --session UUID --output FILE [--workspace PATH]\n
Previously supported: voder model|plugin|doctor|daemon|memory\n`}
async function readOnlyController(flags){const cwd=await workspace(flags.workspace);const journal=await new VoderJournal({dir:flags['session-dir']||envPath(),workspace:cwd}).initialize();return journal;}
export async function main(argv=process.argv.slice(2),{output=process.stdout,error=process.stderr,modelHost}={}){
 const [sub,...rest]=argv;if(!sub||['help','--help'].includes(sub)){output.write(help());return 0;}
 const {flags,pos}=parse(rest);
 if(sub==='run'||sub==='rpc'){
  allowed(flags,['goal','workspace','session-dir','draft','gateway-config','memory','format','web','dialogue-storage-root']);
  if(pos.length)fail('INVALID_ARGUMENT','Unexpected positional argument');
  if(flags.web!==undefined&&flags.web!==true)fail('INVALID_OPTION','--web is a boolean switch');
  if(flags.draft!==undefined&&flags.draft!==true)fail('INVALID_OPTION','--draft is a boolean switch');
  if(sub==='rpc'&&flags.format!==undefined)fail('INVALID_OPTION','RPC has fixed JSON-lines framing');
  const control=await makeControl({cwd:flags.workspace,sessionDir:flags['session-dir'],gate:flags['gateway-config']??process.env.VODER_NATIVE_GATEWAY_CONFIG,
    draft:flags.draft===true,modelHost,web:flags.web===true,dialogueStorageRoot:flags['dialogue-storage-root']});
  if(sub==='rpc'){
   if(flags.goal!==undefined||flags.memory!==undefined)fail('INVALID_OPTION','RPC takes session parameters through protocol, not flags');
   await serveRpc(control,{output});return 0;
  }
  const format=flags.format??'json';if(!['json','text'].includes(format))fail('INVALID_FORMAT','--format must be json or text');
  const goal=str(flags.goal,'--goal');
  const session=await control.createSession({memoryFiles:flags.memory??[]});
  const sig=()=>control.cancel(session.sessionId).catch(()=>{});
  process.on('SIGINT',sig);process.on('SIGTERM',sig);
  try{
   await control.submitTask(session.sessionId,{goal});
   const result=await control.wait(session.sessionId);
   if(format==='text')writeHumanResult(output,result,session.sessionId,control.journal.dir);
   else send(output,{schema:'voder.cli.run.v1',session_id:session.sessionId,...result,
     journal:control.journal.dir});
   return ['VERIFIED','DELIVERED'].includes(result.status)?0:result.status==='DRAFT_UNVERIFIED'?3:2;
  }finally{process.off('SIGINT',sig);process.off('SIGTERM',sig);control.dispose();}
 }
 allowed(flags,['session','workspace','session-dir','after','output','gateway-config']);if(pos.length)fail('INVALID_ARGUMENT','Unexpected positional argument');
 const id=str(flags.session,'--session');const journal=await readOnlyController(flags);
 let result;
 switch(sub){
  case 'status': {const r=await journal.get(id);result={session_id:id,status:r.status,started_at:r.started_at,ended_at:r.ended_at,result:r.result};break;}
  case 'events':{const n=flags.after===undefined?0:Number(flags.after);result=await journal.events(id,{after:n});break;}
  case 'resume':{const r=await journal.get(id);result={session_id:id,status:r.status,mode:'READ_ONLY_REATTACH',canContinue:false,
   reason:'Native VGR reconciliation required before resuming tools',events:r.events};break;}
  case 'diff':{
    const config=await gatewayConfig(flags['gateway-config']??process.env.VODER_NATIVE_GATEWAY_CONFIG);
    result=await readNativeSessionDiff({journal,id,attemptStore:config?.attemptStore,policyFile:config?.policyFile,
      commitVerifier:config?new GatewayCommitVerifier({daemonBinary:config.daemonBinary,attemptStore:config.attemptStore}):undefined});
    break;
  }
  case 'export':result=await journal.export(id,str(flags.output,'--output'));break;
  default:fail('UNKNOWN_COMMAND','Use voder help');
 }
 send(output,result);return 0;
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 process.umask(0o077);
 main().then(rc=>{process.exitCode=rc;}).catch(e=>{process.stderr.write(`VODER_CLI=BLOCKED code=${e?.code??'ERROR'} reason=${String(e?.message??'failure').slice(0,260)}\n`);process.exitCode=1;});
}
