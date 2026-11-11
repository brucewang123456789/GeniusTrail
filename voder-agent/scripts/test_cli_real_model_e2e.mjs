#!/usr/bin/env node
/**
 * Voder CLI real-model acceptance: exercises the exact installed ./bin/voder command,
 * a real Pi provider call, a required repository tool read, live journal, and
 * independent-process status/events/export readers.  NEVER treats draft as VGR commit.
 * Does not modify product source. All task data are synthetic and held in private repair.
 */
import {randomBytes,createHash} from 'node:crypto';
import {mkdtemp,mkdir,writeFile,readFile,lstat,chmod} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';

const TAG='VODER_CLI_REAL_E2E';
const ROOT='/root/autodl-tmp';
const EXPECTED=Object.freeze({
 'bin/voder':'d65c67aa14a87cdf10f5d427d7101d6de9c582bcf4d3f7fa4136f882e788b7eb',
 'voder-agent/packages/voder-cli/bin/voder-cli.mjs':'d5c641e5824e137debf56d7aa7b082d024f2352d6390a0117b5d92ca3b054843',
 'voder-agent/packages/voder-sdk/src/control-plane.mjs':'87b22ee386c4b287fa5b74eaba32d93402e0b3cc506d771b9cd1fbfa7a29af8c',
 'voder-agent/packages/voder-sdk/src/index.mjs':'1e57413ab62c5e75ba340aa9277b32c1237ebf45b9cd6264f52ca68135fa21d0',
 'voder-agent/packages/voder-agent-core/src/kernel.mjs':'bed3095e7acb9fc2863d141a71857f9044e3e4b04643f535c6d3b258b4120e78'
});
const SHA=v=>createHash('sha256').update(v).digest('hex');
const gate=(name,pass,details='')=>{
 console.log(`${TAG}_${name}=${pass?'PASS':'FAIL'}${details?' '+details:''}`);
 if(!pass)throw Object.assign(new Error(`GATE_${name}_FAILED`),{code:`GATE_${name}_FAILED`});
};
function assert(c,msg){if(!c)throw Object.assign(new Error(msg),{code:msg});}
async function exec(args,{maxTime=20000,cwd=ROOT}={}){
 return await new Promise((done,reject)=>{
  const p=spawn(join(ROOT,'bin','voder'),args,{cwd,env:process.env,stdio:['ignore','pipe','pipe'],detached:true});
  let stdout='',stderr='',timedOut=false,overflow=false,finished=false;
  const timer=setTimeout(()=>{
   timedOut=true;try{process.kill(-p.pid,'SIGINT')}catch{}
   setTimeout(()=>{if(!finished)try{process.kill(-p.pid,'SIGKILL')}catch{}},3500).unref();
  },maxTime);
  p.stdout.on('data',chunk=>{stdout+=chunk.toString('utf8');if(Buffer.byteLength(stdout)>512*1024){overflow=true;try{process.kill(-p.pid,'SIGKILL')}catch{}}});
  p.stderr.on('data',chunk=>{stderr+=chunk.toString('utf8');if(Buffer.byteLength(stderr)>128*1024){overflow=true;try{process.kill(-p.pid,'SIGKILL')}catch{}}});
  p.on('error',err=>{if(!finished){finished=true;clearTimeout(timer);reject(err)}});
  p.on('close',(rc,signal)=>{if(finished)return;finished=true;clearTimeout(timer);done({rc,signal,stdout,stderr,timedOut,overflow})});
 });
}
function onlyJson(s){const lines=s.trim().split(/\r?\n/).filter(Boolean);assert(lines.length===1,'CLI_UNEXPECTED_NON_JSON_STDOUT');return JSON.parse(lines[0]);}
async function main(){
 process.umask(0o077);
 assert(process.argv.length===2,'UNEXPECTED_ARGS');
 assert(process.platform==='linux','AUTODL_LINUX_REQUIRED');
 assert(process.env.VODER_MODEL_ID?.trim() && process.env.VODER_MODEL_API_KEY?.trim(),'MODEL_ID_AND_API_KEY_REQUIRED');
 const v=process.versions.node.split('.').map(Number);
 assert(v[0]>22||(v[0]===22&&v[1]>=19),'PI_SUPPORTED_NODE_REQUIRED');
 for(const [path,digest] of Object.entries(EXPECTED)){
  const buffer=await readFile(join(ROOT,path));
  assert(SHA(buffer)===digest,`SOURCE_VERSION_DRIFT:${path}`);
 }
 gate('PINNED_RELEASE',true);
 const scratch=await mkdtemp(join(ROOT,'repair','.voder-cli-real-e2e-'));
 await chmod(scratch,0o700);
 const workspace=join(scratch,'workspace'),journal=join(scratch,'journal');
 await mkdir(workspace,{mode:0o700});await mkdir(journal,{mode:0o700});
 const evidence=join(scratch,'ACCEPTANCE.json');
 console.log('E2E_EVIDENCE_DIR='+scratch);
 const marker='CLI-'+randomBytes(12).toString('hex').toUpperCase();
 const fixture=join(workspace,'blueprint_probe.txt');
 await writeFile(fixture,`# Test-only source; not a user project\nverification_code = ${marker}\ndecoy = NOT_THE_VERIFICATION_CODE\n`,{mode:0o600,flag:'wx'});
 const baseline=SHA(await readFile(fixture));
 const goal='真实编程Agent只读检索验收：必须先调用 voder_read_file 工具读取当前工作区 blueprint_probe.txt 文件，然后从文件中提取 verification_code 的值，仅输出该值。不要猜测，不要根据本提示编造值；不要写入、修改、运行任何命令，也不要声称代码已验证或VGR已提交。';
 console.log(`${TAG}_REAL_MODEL_TASK_BEGIN=YES`);
 const started=Date.now();
 const run=await exec(['run','--workspace',workspace,'--session-dir',journal,'--draft','--goal',goal],{maxTime:240000});
 const duration=Date.now()-started;
 const errorCode=run.stderr.match(/VODER_CLI=BLOCKED code=([A-Z0-9_]+)/)?.[1]??run.stderr.match(/\b(ERR_[A-Z0-9_]+)\b/)?.[1]??null;
 const core={elapsed_ms:duration,cli_exit_code:run.rc,cli_timeout:run.timedOut,cli_error_code:errorCode,scope:'real_CLI_executable_real_Pi_provider_read_tool_journal_cross_process',
  native_vgr_commit_tested:false,vscode_tested:false};
 try{
  gate('MODEL_PROCESS_EXIT',!run.timedOut&&!run.overflow&&run.rc===3,`RC=${run.rc}`);
  const result=onlyJson(run.stdout);
  gate('DRAFT_NOT_FAKE_VERIFIED',result?.status==='DRAFT_UNVERIFIED'&&result?.verified===false&&result?.draft_available===true);
  const id=result.session_id;
  assert(/^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/i.test(id),'BAD_SESSION_ID');
  core.session_id=id;
  const snap=join(journal,'sessions',id,'snapshot.json');
  const snapshotText=await readFile(snap,'utf8');const record=JSON.parse(snapshotText);
  const {sha256,...body}=record;
  gate('JOURNAL_SHA256',sha256===SHA(JSON.stringify(body))&&record.status==='DRAFT_UNVERIFIED');
  const eventTypes=record.events?.map(e=>e.type)??[];
  core.event_types=eventTypes;
  gate('REAL_PI_MODEL_EVENTS',eventTypes.includes('model.agent_start')&&eventTypes.includes('model.agent_end'));
  gate('MODEL_CHOSE_READ_TOOL',eventTypes.includes('planning.evidence')&&eventTypes.includes('model.tool_execution_start')&&eventTypes.includes('model.tool_execution_end'));
  gate('NO_NATIVE_ACTION',!eventTypes.includes('vgr.action_requested')&&!eventTypes.includes('vgr.action_settled'));
  gate('FIXTURE_UNCHANGED',SHA(await readFile(fixture))===baseline);
  gate('JOURNAL_PRIVACY',!snapshotText.includes(marker)&&!snapshotText.includes(process.env.VODER_MODEL_API_KEY));
  const stat=await lstat(snap);
  gate('PRIVATE_JOURNAL_FILE',(stat.mode&0o077)===0);
  const status=await exec(['status','--workspace',workspace,'--session-dir',journal,'--session',id]);
  gate('INDEPENDENT_PROCESS_STATUS',status.rc===0&&onlyJson(status.stdout).status==='DRAFT_UNVERIFIED');
  const history=await exec(['events','--workspace',workspace,'--session-dir',journal,'--session',id]);
  const events=onlyJson(history.stdout);
  gate('INDEPENDENT_PROCESS_EVENTS',history.rc===0&&Array.isArray(events)&&events.length===eventTypes.length);
  const exportPath=join(scratch,'session-metadata.json');
  const exported=await exec(['export','--workspace',workspace,'--session-dir',journal,'--session',id,'--output',exportPath]);
  const receipt=onlyJson(exported.stdout);
  gate('INDEPENDENT_PROCESS_EXPORT',exported.rc===0&&receipt.status==='EXPORTED_UNVERIFIED_SESSION_METADATA'&&receipt.path===exportPath);
  const exportedContent=await readFile(exportPath,'utf8');
  gate('EXPORT_METADATA_ONLY',!exportedContent.includes(marker)&&!exportedContent.includes(process.env.VODER_MODEL_API_KEY));
  const es=await lstat(exportPath);
  gate('PRIVATE_EXPORT_FILE',(es.mode&0o077)===0);
  core.core_pipeline_pass=true;
  console.log(`${TAG}_REAL_MODEL_CLI_CONTROL_PIPELINE=PASS`);
  // Correctness requires checking exact model text in the *live CLI result*.
  // Persistent journals and exported sessions remain metadata only.
  const visibleAnswer=typeof result.draft==='string'?result.draft:
   typeof result.answer==='string'?result.answer:
   typeof result.content==='string'?result.content:null;
  const delivered=typeof visibleAnswer==='string'&&visibleAnswer.trim()===marker && result.output_authority==='UNVERIFIED_MODEL_TEXT_NOT_VGR_PROOF';
  core.model_answer_delivered=delivered;
  core.answer_visibility=visibleAnswer===null?'CLI_RESULT_DOES_NOT_INCLUDE_MODEL_ANSWER':delivered?'MATCH':'ANSWER_MISMATCH';
  core.result=delivered?'PASS':'PARTIAL_BLOCKED';
  await writeFile(evidence,JSON.stringify(core,null,2)+'\n',{mode:0o600,flag:'wx'});
  console.log(`ACCEPTANCE_REPORT=${evidence}`);
  if(delivered)console.log(`${TAG}_USER_VISIBLE_ANSWER=PASS authority=${result.output_authority??'MISSING'}`);
  if(!delivered){
   console.log(`${TAG}_USER_VISIBLE_ANSWER=BLOCKED reason=${core.answer_visibility}`);
   console.log(`${TAG}_FULL_TASK_E2E=NOT_PASSED`);
   process.exitCode=4;
  }else console.log(`${TAG}_FULL_TASK_E2E=PASS`);
 }catch(e){
  core.result='FAIL';core.failure_code=e.code??'ASSERTION_FAILED';
  await writeFile(evidence,JSON.stringify(core,null,2)+'\n',{mode:0o600,flag:'wx'}).catch(()=>{});
  console.error(`${TAG}=FAIL code=${core.failure_code} CLI_RC=${run.rc} CLI_ERROR_CODE=${errorCode??'UNKNOWN'} evidence=${evidence}`);
  process.exitCode=1;
 }
}
main().catch(e=>{console.error(`${TAG}=PRECHECK_FAIL code=${e.code||e.message||'ERROR'}`);process.exitCode=1;});
