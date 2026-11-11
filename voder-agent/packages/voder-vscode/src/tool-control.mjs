/** Opt-in multi-file RPC adapter. Reuses unchanged SDK, journal, native transport and RPC framing. */
import {pathToFileURL,fileURLToPath} from 'node:url';
import {resolve,join,dirname} from 'node:path';
import {readFile,realpath} from 'node:fs/promises';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
const {validateExtended}=require('./multi-workspace.cjs');
const {readScope,inspectSourceRows,inspectDependencies,hash}=require('./workspace-tools.cjs');
import {SyntaxThenSandboxVerifier,privateDiagnosticWriter} from './candidate-control.mjs';
export {SyntaxThenSandboxVerifier} from './candidate-control.mjs';
export async function makeToolkitControl({cwd,sessionDir,gate,agentRoot,modelHost}={}){
 for(const p of [cwd,gate,agentRoot,sessionDir])if(typeof p!=='string'||!p.startsWith('/'))throw Error('TOOLKIT_ABSOLUTE_HOST_PATHS_REQUIRED');
 if(await realpath(agentRoot)!==agentRoot||await realpath(cwd)!==cwd)throw Error('TOOLKIT_CANONICAL_ROOT_REQUIRED');
 const storage=resolve(dirname(gate),'../..'),project=await validateExtended(join(dirname(gate),'project.json'),storage);
 if(project.root!==cwd||project.profile!==gate)throw Error('TOOLKIT_PROJECT_IDENTITY');
 const load=relative=>import(pathToFileURL(join(agentRoot,relative)).href);
 const [{VoderSDK,VoderControlPlane,VoderJournal},{HostCandidateVerifier},{NativeSdkContinuation},{HostToolRegistry}]=await Promise.all([load('packages/voder-sdk/src/index.mjs'),load('packages/voder-sdk/src/candidate-verifier.mjs'),load('packages/voder-sdk/src/native-continuation.mjs'),load('packages/voder-tools/src/index.mjs')]);
 const cfg=JSON.parse(await readFile(gate,'utf8')),base=new HostCandidateVerifier({cwd,nativeGateway:cfg}),candidateVerifier=new SyntaxThenSandboxVerifier(base,{targets:project.targets,node:project.nodeBinary,report:privateDiagnosticWriter(cfg.attemptStore)}),registry=new HostToolRegistry();
 const empty={type:'object',properties:{},required:[],additionalProperties:false};
 registry.register({name:'voder_ext_project_syntax',description:'Parse current approved workspace files without executing code. These are PREIMAGE observations, not a candidate PASS or delivery proof. Candidate syntax is independently checked before sandbox verification.',capability:'verify',parameters:empty,execute:async()=>({authority:'PASSIVE_PREIMAGE_OBSERVATIONS',checks:await inspectSourceRows(await readScope(cwd,project.targets),{node:project.nodeBinary})})});
 registry.register({name:'voder_ext_project_dependencies',description:'Read dependency declarations in approved files. Does not install packages, run scripts, start services or grant native permissions.',parameters:empty,execute:async()=>inspectDependencies(await readScope(cwd,project.targets))});
 registry.register({name:'voder_ext_project_capabilities',description:'Read exact host-installed capabilities and known limitations. No subprocess or native action.',parameters:empty,execute:async()=>({scope:[...project.targets],candidate_contract:candidateVerifier.describe(),parsers:['JavaScript syntax','Python AST','JSON'],candidate_gate:'passive syntax followed by unchanged pinned Modal validator and native VGR',browser:'post-delivery bounded untrusted observations; no automated visual oracle',package_installation:false,arbitrary_shell:false})});
 require('./dialogue-model-tool.cjs').registerDialogueModelTool(registry,{projectRoot:cwd,storageRoot:storage});
 const sdk=new VoderSDK({cwd,nativeGateway:cfg,candidateVerifier,toolRegistry:registry,toolEconomyOptions:{maxCalls:6,maxFailuresPerTool:1,maxReturnBytes:131072,maxSingleReturnBytes:32768},modelProfiles:process.env.VODER_MODEL_PROFILES_JSON?JSON.parse(process.env.VODER_MODEL_PROFILES_JSON):[],...(modelHost?{modelHost}:{})});
 await sdk.preflight();base.onInfrastructureError=id=>sdk.cancel(id);
 const journal=new VoderJournal({dir:sessionDir,workspace:cwd});await journal.initialize();
 return new VoderControlPlane({sdk,journal,nativeContinuation:new NativeSdkContinuation({gatewayBinary:cfg.binary,daemonBinary:cfg.daemonBinary,attemptStore:cfg.attemptStore})});
}
async function main(){
 const [sub,...args]=process.argv.slice(2);if(sub!=='rpc')throw Error('TOOLKIT_RPC_ONLY');const flags={};
 for(let i=0;i<args.length;i+=2){if(!['--workspace','--session-dir','--gateway-config','--agent-root'].includes(args[i])||typeof args[i+1]!=='string'||Object.hasOwn(flags,args[i]))throw Error('TOOLKIT_INVALID_FLAGS');flags[args[i]]=args[i+1];}
 process.umask(0o077);
 const control=await makeToolkitControl({cwd:flags['--workspace'],sessionDir:flags['--session-dir'],gate:flags['--gateway-config'],agentRoot:flags['--agent-root']});
 const {serveRpc}=await import(pathToFileURL(join(flags['--agent-root'],'packages/voder-cli/bin/voder-cli.mjs')).href);await serveRpc(control);
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url))main().catch(e=>{process.stderr.write('VODER_TOOLKIT_RPC_BLOCKED='+String(e.code||'CONTROL_START_FAILED')+'\n');});
