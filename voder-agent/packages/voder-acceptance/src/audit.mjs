#!/usr/bin/env node
/** Evidence-based offline audit; local test PASS is never promoted to end-to-end PASS. */
import {readFile,access} from 'node:fs/promises';
import {resolve,join} from 'node:path';
import {moduleRoot,sha,readJson,fileHash,privateDir,persist,processRun,taskRunId,time,safeRuntimeEnv,requireThat} from './common.mjs';
const agent=resolve(process.env.VODER_AGENT_ROOT||join(moduleRoot,'../..'));
const repo=resolve(agent,'..');
const output=resolve(process.env.VODER_M8_EVIDENCE_ROOT||join(repo,'.voder-build','acceptance'));
const gateNames=['A01 Required files','A02 System prompt','A03 Dynamic tools','A04 Skills','A05 Repair','A06 Native commit','A07 Failure/recovery','A08 Search & memory','A09 SDK CLI','A10 VS Code','A11 Isolation','A12 Metrics'];
const gates=Object.fromEntries(gateNames.map((x,i)=>['A'+String(i+1).padStart(2,'0'),{name:x,status:'NOT_VERIFIED',evidence:[]}]));
const add=(key,status,detail)=>{gates[key].status=status;gates[key].evidence.push(detail)};
async function main(){
 const id=taskRunId(),run=await privateDir(join(output,id));
 const report={schema:'voder.module8.audit.v1',run_id:id,started_at:time(),agent_root:agent,mode:'OFFLINE_NO_MODEL_NO_MODAL',node:process.version,gates,checks:[],original_source_mutated:false,production_candidate:false};
 let failures=0, blockers=0;
 const push=(label,pass,detail)=>{report.checks.push({label,status:pass?'PASS':'FAIL',detail});if(!pass)failures++};
 try{
  const lock=await readJson(join(moduleRoot,'policies','source-lock.json'));
  // A09/A10 is an approved, hash-pinned additive overlay: each changed original
  // must have an immutable before-image verified against the original 909-file lock.
  // No other frozen source difference is permitted.
  const receiptPath=join(repo,'repair','voder-module8-overlays','active.json');
  let overlay=null;try{overlay=await readJson(receiptPath)}catch{}
  const approved=lock.approved_overrides??{};
  let missing=0,changed=0,ok=0,approvedMatch=0;
  for(const [p,expected] of Object.entries(lock.protected_files)){
   let actual;try{actual=await fileHash(join(repo,p))}catch{missing++;continue;}
   if(actual===expected){ok++;continue;}
   const spec=approved[p],backup=overlay?.backups?.[p];
   if(!spec||spec.original_sha256!==expected||actual!==spec.installed_sha256||
      overlay?.schema!=='voder.module8.overlay.provenance.v1'||
      typeof backup!=='string'||!backup.startsWith(join(repo,'repair','voder-module8-backups')+'/')){
     changed++;continue;
   }
   try{if(await fileHash(backup)===expected){approvedMatch++;continue}}catch{}
   changed++;
  }
  report.source_integrity={expected:Object.keys(lock.protected_files).length,original_match:ok,
    approved_overrides:approvedMatch,match:ok+approvedMatch,missing,mismatched:changed,
    archive_sha256:lock.archive_sha256,originals_backup_provenance:approvedMatch===Object.keys(approved).length};
  push('LOCKED_FROZEN_1_TO_7',missing===0&&changed===0&&approvedMatch===Object.keys(approved).length,report.source_integrity);
  if(missing||changed||approvedMatch!==Object.keys(approved).length){
    add('A06','BLOCKED_BASELINE','Unapproved source drift or missing original-before-image backup');
    blockers++;throw Object.assign(new Error('BASELINE_MISMATCH_TESTS_ABORTED'),{code:'BASELINE_MISMATCH_TESTS_ABORTED'});
  }
  const manifest=await readJson(join(moduleRoot,'fixtures','acceptance-tasks.json'));
  push('FIXED_TASK_FIXTURE_IDENTITY',sha(JSON.stringify(manifest))===lock.task_manifest_sha256,{count:manifest.tasks.length,sha256:sha(JSON.stringify(manifest))});
  const src=await readFile(join(agent,'packages/voder-sdk/src/control-plane.mjs'),'utf8');
  const cli=await readFile(join(agent,'packages/voder-cli/bin/voder-cli.mjs'),'utf8');
  const hasDiff=src.includes('readNativeSessionDiff') && cli.includes('readNativeSessionDiff');
  const vsCode=await readFile(join(agent,'packages/voder-vscode/src/extension.cjs'),'utf8');
  const hasReview=vsCode.includes("registerTextDocumentContentProvider('voder-evidence'") && vsCode.includes("'vscode.diff'");
  add('A09',hasDiff?'OFFLINE_INTEGRATED_LIVE_PENDING':'BLOCKED_CAPABILITY',
    hasDiff?'SDK/RPC/CLI native diff implemented; independent Rust+real session still required':'Session-scoped SDK/CLI diff missing');
  add('A10',hasReview&&hasDiff?'OFFLINE_INTEGRATED_LIVE_PENDING':'BLOCKED_CAPABILITY',
    hasReview&&hasDiff?'VS Code readonly diff provider integrated; extension host and native approval workflow not proven':'Editor review integration missing');
  if(!hasDiff||!hasReview)blockers++;

  const files=[...['voder-agent-core','voder-context','voder-task-planner','voder-tools','voder-skills','voder-diagnostics','voder-native-bridge','voder-search','voder-memory','voder-sdk','voder-orchestration'].map(x=>join(agent,'packages',x,'test'))];
  const {readdir}=await import('node:fs/promises');const tests=[];
  for(const d of files){let a;try{a=await readdir(d)}catch{continue}for(const f of a.filter(f=>f.endsWith('.test.mjs')))tests.push(join(d,f))}
  for(const f of (await readdir(join(agent,'packages/voder-vscode/test'))).filter(f=>f.endsWith('.test.cjs')))tests.push(join(agent,'packages/voder-vscode/test',f));
  const fixtureRoot=join(agent,'packages','voder-vscode','test','fixtures');
  const env={...safeRuntimeEnv(),VODER_AGENT_TEST_ROOT:agent,
    VODER_ARCHIVE_FIXTURES:join(fixtureRoot,'archives'),VODER_DOCUMENT_FIXTURES:join(fixtureRoot,'documents'),
    VODER_PDF_VISION_FIXTURES:join(fixtureRoot,'documents'),
    VODER_NATIVE_TEST_GATEWAY:join(repo,'.voder-build','release','voder-gateway')};
  const testResult=await processRun(process.execPath,['--test','--test-reporter=tap','--test-concurrency=2',...tests],{cwd:agent,timeoutMs:600000,env});
  const tap=testResult.stdout;const lastNum=(name)=>{const m=[...tap.matchAll(new RegExp('^# '+name+' (\\d+)$','gm'))];return m.length?Number(m.at(-1)[1]):null};
  report.node_runtime_warning=/^v22\.(1[0-8])\./.test(process.version)?'Node runtime is below upstream Pi >=22.19 requirement':null;
  report.offline_tests={file_count:tests.length,test_count:lastNum('tests'),pass:lastNum('pass'),fail:lastNum('fail'),skipped:lastNum('skipped'),timeout:testResult.timeout,duration_ms:testResult.duration_ms,exit_code:testResult.exit_code,stdout_sha256:sha(testResult.stdout),stderr_sha256:sha(testResult.stderr)};
  push('OFFLINE_TESTS',testResult.exit_code===0&&!testResult.timeout&&lastNum('tests')>0&&lastNum('fail')===0,report.offline_tests);
  for(const k of ['A01','A02','A03','A04','A05','A07','A08','A11'])add(k,'OFFLINE_TESTED_NOT_LIVE',`Offline test suite ${report.offline_tests.test_count??0} tests; no actual model + Rust/Modal obligation proved`);
  add('A06','NOT_VERIFIED','Rust native real authorization+journal+workspace delivery requires paid, trusted live environment');
  add('A12','NOT_VERIFIED','Frozen live suite must be executed with actual model and collect tokens/cost/latency and failure distribution');
  let postchanged=0;for(const [p,expected] of Object.entries(lock.protected_files)){
    try{const got=await fileHash(join(repo,p));if(got!==expected&&got!==approved[p]?.installed_sha256)postchanged++;}
    catch{postchanged++}
  }
  report.post_test_integrity={rechecked:Object.keys(lock.protected_files).length,drifted:postchanged};
  push('POST_TEST_FROZEN_SOURCE_UNCHANGED',postchanged===0,report.post_test_integrity);
  if(failures)blockers++;
  report.outcome=blockers?'BLOCKED':'READY_FOR_LIVE_NOT_PRODUCTION';report.blockers=blockers;report.failures=failures;report.ended_at=time();
 }catch(e){report.outcome='BLOCKED';report.error_code=e.code??'AUDIT_ERROR';report.error=String(e.message).slice(0,280);report.ended_at=time();}
 await persist(join(run,'AUDIT.json'),report);
 console.log(`VODER_M8_AUDIT=${report.outcome} checks=${report.checks.filter(x=>x.status==='PASS').length}/${report.checks.length} evidence=${join(run,'AUDIT.json')}`);
 console.log(`VODER_M8_SOURCE=${JSON.stringify(report.source_integrity??{})}`);
 console.log(`VODER_M8_OFFLINE=${JSON.stringify(report.offline_tests??{})}`);
 console.log(`VODER_M8_CAPABILITY_BLOCKERS=${JSON.stringify(['A09','A10'].map(k=>({gate:k,status:gates[k].status})))}`);
 process.exitCode=report.outcome==='BLOCKED'?4:0;
}
main().catch(e=>{console.error('VODER_M8_AUDIT_FATAL='+String(e.message).slice(0,180));process.exitCode=5});
