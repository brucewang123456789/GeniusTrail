/** General task RPC: same SDK, sandbox receipts, Rust authority and native delivery. */
import {pathToFileURL,fileURLToPath} from 'node:url';
import {resolve,join,dirname} from 'node:path';
import {readFile,realpath} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
const {validateTask}=require('./task-workspace.cjs');
const {inspectSourceRows,hash}=require('./workspace-tools.cjs');
export class TaskCandidateVerifier {
 constructor(base,{targets,node}={}){this.base=base;this.targets=targets;this.node=node;}
 getBudget(id){return this.base.getBudget(id);}
 async verifyCandidate(candidate,options={}){
  const members=candidate?.patches??[candidate];
  if(JSON.stringify(members.map(p=>p?.relative_path))!==JSON.stringify(this.targets))throw Object.assign(Error('Candidate must cover exactly the approved sorted files.'),{code:'TASK_CANDIDATE_SCOPE'});
  if(members.some(p=>typeof p.new_content!=='string'||Buffer.byteLength(p.new_content)<1||p.new_content.includes('\0')))throw Object.assign(Error('Candidate must be nonempty NUL-free UTF-8 text.'),{code:'TASK_CANDIDATE_SIZE'});
  const checks=await inspectSourceRows(members.map(p=>({relative:p.relative_path,bytes:Buffer.from(p.new_content),sha256:hash(p.new_content)})),{node:this.node});
  if(checks.some(c=>c.status==='UNAVAILABLE'))throw Object.assign(Error('Required host parser unavailable; no independent sandbox check started.'),{code:'TASK_PARSER_UNAVAILABLE'});
  const failed=checks.filter(c=>c.status==='FAIL');
  if(failed.length)return{status:'FAIL',backend:'host-passive-parser',sandboxed:false,checks:failed.map(c=>({name:c.file,passed:false})),diagnostic:failed.map(c=>c.file+': '+c.diagnostic).join('\n').slice(-4000)};
  return this.base.verifyCandidate(candidate,options);
 }
}
export async function makeTaskControl({cwd,sessionDir,gate,agentRoot,modelHost}={}){
 for(const p of [cwd,gate,agentRoot,sessionDir])if(typeof p!=='string'||!p.startsWith('/'))throw Error('TASK_ABSOLUTE_HOST_PATHS_REQUIRED');
 if(await realpath(agentRoot)!==agentRoot||await realpath(cwd)!==cwd)throw Error('TASK_CANONICAL_ROOT_REQUIRED');
 const storage=resolve(dirname(gate),'../..'),project=await validateTask(join(dirname(gate),'project.json'),storage);
 if(project.root!==cwd||project.profile!==gate)throw Error('TASK_PROJECT_IDENTITY');
 const load=relative=>import(pathToFileURL(join(agentRoot,relative)).href);
 const [{VoderSDK,VoderControlPlane,VoderJournal},{HostCandidateVerifier},{NativeSdkContinuation},{HostToolRegistry}]=await Promise.all([load('packages/voder-sdk/src/index.mjs'),load('packages/voder-sdk/src/candidate-verifier.mjs'),load('packages/voder-sdk/src/native-continuation.mjs'),load('packages/voder-tools/src/index.mjs')]);
 const cfg=JSON.parse(await readFile(gate,'utf8'));
 const base=new HostCandidateVerifier({cwd,nativeGateway:cfg,timeoutMs:180000}),candidateVerifier=new TaskCandidateVerifier(base,{targets:project.targets,node:project.nodeBinary}),registry=new HostToolRegistry();
 registry.register({name:'voder_ext_task_capabilities',description:'Exact approved task scope, sandbox checks and runtime limits. A capability inventory is not test evidence or a grant of additional permissions.',parameters:{type:'object',properties:{},required:[],additionalProperties:false},execute:async()=>({scope:project.targets,checks:project.checks,source_bytes_per_file:null,total_source_bytes:null,command_timeout_seconds:20,execution:'reviewed argv inside independent sandbox only',commit:'unchanged Rust VGR + native verified workspace delivery',visual_inspection:'UNAVAILABLE',binary_delivery:'UNAVAILABLE',package_installation:'not automatically authorized',verification:'configured checks only; never a visual or task-correctness certificate'})});
 registry.register({name:'voder_ext_task_content_hash',description:'Compute exact SHA256 and UTF-8 size of proposed text WITHOUT writing, validating or committing it. Use the unchanged final source bytes when recording artifact hashes. This is a byte hash, not proof of quality or delivery.',parameters:{type:'object',properties:{content:{type:'string',description:'Exact proposed UTF-8 source; no whole-file byte ceiling'}},required:['content'],additionalProperties:false},execute:async({content})=>({sha256:createHash('sha256').update(content,'utf8').digest('hex'),bytes:Buffer.byteLength(content,'utf8'),written:false,verified:false})});
 require('./dialogue-model-tool.cjs').registerDialogueModelTool(registry,{projectRoot:cwd,storageRoot:storage});
 const sdk=new VoderSDK({cwd,nativeGateway:cfg,candidateVerifier,toolRegistry:registry,toolEconomyOptions:{maxCalls:12,maxFailuresPerTool:2,maxReturnBytes:131072,maxSingleReturnBytes:32768},modelProfiles:process.env.VODER_MODEL_PROFILES_JSON?JSON.parse(process.env.VODER_MODEL_PROFILES_JSON):[],...(modelHost?{modelHost}:{})});
 await sdk.preflight();base.onInfrastructureError=id=>sdk.cancel(id);
 const journal=new VoderJournal({dir:sessionDir,workspace:cwd});await journal.initialize();
 return new VoderControlPlane({sdk,journal,nativeContinuation:new NativeSdkContinuation({gatewayBinary:cfg.binary,daemonBinary:cfg.daemonBinary,attemptStore:cfg.attemptStore})});
}
async function main(){
 const [sub,...args]=process.argv.slice(2);if(sub!=='rpc')throw Error('TASK_RPC_ONLY');const flags={};
 for(let i=0;i<args.length;i+=2){if(!['--workspace','--session-dir','--gateway-config','--agent-root'].includes(args[i])||typeof args[i+1]!=='string'||Object.hasOwn(flags,args[i]))throw Error('TASK_INVALID_FLAGS');flags[args[i]]=args[i+1];}
 process.umask(0o077);
 const control=await makeTaskControl({cwd:flags['--workspace'],sessionDir:flags['--session-dir'],gate:flags['--gateway-config'],agentRoot:flags['--agent-root']});
 const {serveRpc}=await import(pathToFileURL(join(flags['--agent-root'],'packages/voder-cli/bin/voder-cli.mjs')).href);await serveRpc(control);
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url))main().catch(e=>{process.stderr.write('VODER_TASK_RPC_BLOCKED='+String(e.code||'CONTROL_START_FAILED')+'\n');process.exitCode=1;});
