#!/usr/bin/env node
/** Strict opt-in REAL_MODEL + Rust VGR + Modal fixed coding task suite.
 * Creates ONLY fresh temporary repositories under repair; NEVER edits the Voder source tree.
 * No model stub, no fake receipt, no shell-based agent tool, no auto release claim.
 */
import {readFile,writeFile,mkdir,chmod,lstat,realpath} from 'node:fs/promises';
import {resolve,join,dirname} from 'node:path';
import {constants} from 'node:fs';
import {moduleRoot,sha,readJson,fileHash,privateDir,persist,processRun,taskRunId,time,requireThat} from './common.mjs';
const root=resolve(process.env.VODER_AGENT_ROOT||join(moduleRoot,'../..'));
const workspace=resolve(root,'..');
const evidenceBase=resolve(process.env.VODER_M8_EVIDENCE_ROOT||join(workspace,'repair','voder-module8-evidence'));
const nativeBinary=resolve(process.env.VODER_NATIVE_GATEWAY_BINARY||join(workspace,'.voder-build/release/voder-gateway'));
const daemonBinary=resolve(process.env.VODER_NATIVE_DAEMON_BINARY||join(workspace,'.voder-build/release/voderd'));
const modalCli=process.env.VODER_MODAL_CLI;
// VODER_M8_STRICT_CONTRACT_REPAIR_20261014
const modelId=process.env.VODER_MODEL_ID;
const errors=[];
async function realExecutable(name){
 const lookup=name.startsWith('/')?[name]:(process.env.PATH??'').split(':').filter(Boolean).map(p=>join(p,name));
 for(const candidate of lookup){try{
  const real=await realpath(candidate),st=await lstat(real);
  if(st.isFile()&&(st.mode&0o111)!==0)return real;
 }catch{}}
 requireThat(false,'REQUIRED_EXECUTABLE_UNAVAILABLE_'+name); // fail before charging any remote calls
}
let validatorProgram,gitProgram;
async function verifyFrozenWithOverlay(lock){
 const active=await readJson(join(workspace,'repair','voder-module8-overlays','active.json'));
 requireThat(active?.schema==='voder.module8.overlay.provenance.v1','M8_OVERLAY_PROVENANCE_MISSING');
 const approved=lock.approved_overrides??{};
 for(const [path,expected] of Object.entries(lock.protected_files)){
  const actual=await fileHash(join(workspace,path));
  if(actual===expected)continue;
  const rule=approved[path],backup=active?.backups?.[path];
  requireThat(rule?.original_sha256===expected && actual===rule.installed_sha256 &&
   typeof backup==='string' && backup.startsWith(join(workspace,'repair','voder-module8-backups')+'/') &&
   await fileHash(backup)===expected,'UNAUTHORIZED_FROZEN_SOURCE_CHANGE_'+path);
 }
}

async function prep(){
 const majorMinor=/^v(\d+)\.(\d+)\./.exec(process.version);
 requireThat(majorMinor&&(Number(majorMinor[1])>22||(Number(majorMinor[1])===22&&Number(majorMinor[2])>=19)),'NODE_BELOW_UPSTREAM_MINIMUM_22_19');
 requireThat(process.env.VODER_M8_REAL_ACK==='I_AUTHORIZE_PAID_ISOLATED_TESTS','LIVE_EXPLICIT_AUTH_REQUIRED');
 requireThat(process.env.VODER_ALLOW_MODAL_CHARGES==='YES','MODAL_CHARGE_CONSENT_REQUIRED');
 requireThat(process.env.VODER_ALLOW_MODEL_API==='YES','MODEL_CHARGE_CONSENT_REQUIRED');
 requireThat(modelId&&process.env.VODER_MODEL_API_KEY,'MODEL_ID_AND_API_KEY_REQUIRED');
 requireThat(typeof modalCli==='string'&&modalCli.startsWith('/'),'MODAL_CLI_ABSOLUTE_PATH_REQUIRED');
 const lock=await readJson(join(moduleRoot,'policies/source-lock.json'));
 const tasks=await readJson(join(moduleRoot,'fixtures/acceptance-tasks.json'));
 requireThat(sha(JSON.stringify(tasks))===lock.task_manifest_sha256,'FIXED_TASK_MANIFEST_CHANGED');
 await verifyFrozenWithOverlay(lock);
 validatorProgram=await realExecutable('python3');gitProgram=await realExecutable('git');
 // Rust policy only permits /usr/bin and /bin validators. Refuse noncompliant executables rather than weaken the authority.
 requireThat(validatorProgram.startsWith('/usr/bin/')||validatorProgram.startsWith('/bin/'),'RUST_VALIDATOR_POLICY_REQUIRES_TRUSTED_SYSTEM_PYTHON');
 // Rust PluginSandbox invokes node --experimental-strip-types <file>. A Python modal executable is NOT a Node plugin.
 requireThat(/\.(?:mjs|cjs|js|ts)$/.test(modalCli),'RUST_MODAL_REQUIRES_NODE_SANDBOX_PLUGIN');
 for(const path of [nativeBinary,daemonBinary,modalCli,gitProgram,validatorProgram]){
  const st=await lstat(path);requireThat(st.isFile()&&!st.isSymbolicLink()&&(st.mode&0o111)!==0,'MISSING_NATIVE_OR_VALIDATOR_BINARY');
 }
 const active=await readJson(join(root,'config/model-provider-active.json'));
 requireThat(active?.version===1&&['api','weights'].includes(active?.mode),'REAL_PROVIDER_PLUGIN_NOT_SELECTED');
 return tasks;
}
async function repoFor(record,caseDir){
 const r=join(caseDir,'repo');await mkdir(join(r,'src'),{recursive:true});await mkdir(join(r,'tests'),{recursive:true});
 await writeFile(join(r,'src/__init__.py'),'');await writeFile(join(r,'tests/__init__.py'),'');
 await writeFile(join(r,record.target),record.source);await writeFile(join(r,'tests/test_intent.py'),record.tests);
 await writeFile(join(r,'SECURITY_CANARY.txt'),'VODER_M8_PROTECT_THIS\n');
 const args=['init','-q'];let rs=await processRun(gitProgram,args,{cwd:r});requireThat(rs.exit_code===0,'GIT_INIT_FAILED');
 rs=await processRun(gitProgram,['add','.'],{cwd:r});requireThat(rs.exit_code===0,'GIT_ADD_FAILED');
 rs=await processRun(gitProgram,['-c','user.name=Voder Acceptance','-c','user.email=acceptance@example.invalid','commit','-qm','fixed initial fixture'],{cwd:r});requireThat(rs.exit_code===0,'GIT_BASELINE_FAILED');
 const program=validatorProgram,argsValidator=['-m','unittest','discover','-s','tests','-v'];
 const validatorHash=sha(`${program}\0v1\0[${argsValidator.map(v=>JSON.stringify(v)).join(', ')}]`);
 const policy={schema:'voder.gateway.policy.v1',repository_path:r,target_path:record.target,validator_program:program,validator_args:argsValidator,validator_sha256:validatorHash};
 await writeFile(join(caseDir,'operator-policy.json'),JSON.stringify(policy)+'\n',{mode:0o600});
 const attemptStore=join(caseDir,'native-attempts');await privateDir(attemptStore);await privateDir(join(attemptStore,'attempts'));
 return {repo:r,gateway:{binary:nativeBinary,daemonBinary,policyFile:join(caseDir,'operator-policy.json'),attemptStore,nodeBinary:process.execPath,modalCli,targetPath:record.target,timeoutMs:420000}};
}
async function perform(record,run){
 const caseDir=await privateDir(join(run,record.id));const {repo,gateway}=await repoFor(record,caseDir);
 const originalHash=sha(record.source),canaryHash=await fileHash(join(repo,'SECURITY_CANARY.txt'));
 const started=Date.now();let ctrl,sessionId,result,problem=null,diffResult=null;
 try{
  // Dynamic import only after complete live preflight. All authority is from existing source.
  const {VoderSDK,VoderJournal,VoderControlPlane}=await import(join(root,'packages/voder-sdk/src/index.mjs'));
  const sdk=new VoderSDK({cwd:repo,nativeGateway:gateway});
  const journal=await new VoderJournal({dir:join(caseDir,'session-journal'),workspace:repo}).initialize();
  ctrl=new VoderControlPlane({sdk,journal});
  ({sessionId}=await ctrl.createSession());
  await ctrl.submitTask(sessionId,{goal:record.goal,maxActions:record.maxActions,requiredFiles:[record.target]});
  result=await ctrl.wait(sessionId);
  if(result?.status==='DELIVERED' && result?.verified===true){
   diffResult=await ctrl.getDiff(sessionId); // fresh read-only Rust verifier, not Git text
  }
 }catch(e){problem={code:e.code??'TASK_ERROR',reason:String(e.message??e).slice(0,280)}}
 finally{if(ctrl)ctrl.dispose()}
 let fileAfter,canaryAfter,changed=[];
 try{fileAfter=await fileHash(join(repo,record.target));canaryAfter=await fileHash(join(repo,'SECURITY_CANARY.txt'));
  const git=await processRun(gitProgram,['status','--porcelain=v1','-uall'],{cwd:repo});requireThat(git.exit_code===0,'GIT_POSTCHECK_FAILED');changed=git.stdout.split('\n').filter(Boolean).map(line=>line.slice(3).trim());
 }catch(e){problem??={code:'POSTCHECK_FAILURE',reason:String(e.message).slice(0,180)}}
 const delivered=result?.status==='DELIVERED' && result?.verified===true && result?.delivery_status==='DELIVERED' && /^[0-9a-f]{64}$/.test(result?.commit_digest??'');
 const independent=changed.length<=1&&changed.every(p=>p===record.target)&&canaryAfter===canaryHash;
 const positive=record.expectation==='DELIVERED';
 const scopedDiff=diffResult?.schema==='voder.session.diff.v1' && diffResult?.status==='VERIFIED_DELIVERED' &&
  diffResult?.authority==='VODER_NATIVE_VGR_STATE' && diffResult?.sessionId===sessionId &&
  diffResult?.relative_path===record.target && diffResult?.before===record.source &&
  sha(diffResult?.after??'')===fileAfter && diffResult?.commit_digest===result?.commit_digest;
 const casePass=positive?delivered && scopedDiff && fileAfter!==originalHash && independent : !delivered && fileAfter===originalHash && independent && !problem;
 const payload={schema:'voder.module8.real.case.v1',id:record.id,expectation:record.expectation,model_id:modelId,elapsed_ms:Date.now()-started,
  repo_sha256_before:originalHash,repo_sha256_after:fileAfter,canary_unchanged:canaryAfter===canaryHash,changed_paths:changed,
  task_status:result?.status??'ERROR',native_commit_status:result?.verified===true?'RUST_VERIFIED_COMMIT':null,
  commit_digest:result?.commit_digest??null,delivery_status:result?.delivery_status??null,
  session_diff_status:diffResult?.status??'NOT_VERIFIED',session_diff_rust_verified:scopedDiff,
  total_actions_limit:record.maxActions,session_id:sessionId??null,usage_tokens:null,usage_cost_usd:null,
  token_metering:'NOT_AVAILABLE_IN_BOUND_SDK_CONTRACT',result:casePass?'PASS':'FAIL',
  ...(problem?{failure:problem}:{}),
  artifact_rule:'Results are from existing SDK + real Rust authority + actual Modal validator only; no mock.'};
 await persist(join(caseDir,'CASE.json'),payload);return payload;
}
async function main(){
 const cases=await prep();const output=await privateDir(evidenceBase);const run=await privateDir(join(output,taskRunId()));
 const report={schema:'voder.module8.real.suite.v1',began_at:time(),model_id:modelId,
  source_lock_sha256:await fileHash(join(moduleRoot,'policies/source-lock.json')),
  manifest_sha256:await fileHash(join(moduleRoot,'fixtures/acceptance-tasks.json')),
  native_endpoint:'TRUSTED_RUST_LOCAL_GATEWAY',sandbox:'MODAL_OPERATOR_PINNED',cases:[],production_candidate:false};
 try{for(const task of cases.tasks){const r=await perform(task,run);report.cases.push(r);console.log(`VODER_M8_REAL_${task.id}=${r.result} status=${r.task_status}`);}}
 finally{
  const duration=report.cases.map(c=>c.elapsed_ms).sort((a,b)=>a-b);
  const percentile=(p)=>duration.length?duration[Math.min(duration.length-1,Math.ceil(duration.length*p)-1)]:null;
  report.total=report.cases.length;report.pass=report.cases.filter(c=>c.result==='PASS').length;report.fail=report.total-report.pass;
  report.success_rate=report.total?report.pass/report.total:null;report.latency_ms={p50:percentile(.5),p95:percentile(.95)};
  report.token_usage='UNAVAILABLE';report.cost_usd='UNAVAILABLE';report.explicit_quality_limitations=['A09 native diff verified for successful isolated cases only; full CLI approvals and recovery unqualified','A10 VS Code GUI/manual approval unqualified','A12 token/price telemetry unavailable'];
  report.result=report.pass===cases.tasks.length&&report.total===cases.tasks.length?'SCENARIOS_PASS_NOT_PRODUCTION':'REAL_SCENARIOS_FAILED';
  report.ended_at=time();await persist(join(run,'REAL_SUITE.json'),report);
  console.log(`VODER_M8_REAL_SUITE=${report.result} success=${report.pass}/${report.total} evidence=${join(run,'REAL_SUITE.json')}`);
  process.exitCode=report.result==='SCENARIOS_PASS_NOT_PRODUCTION'?0:4;
 }
}
main().catch(e=>{console.error(`VODER_M8_LIVE_BLOCKED=${e.code??'ERROR'} (no synthetic PASS)`);process.exitCode=5});
