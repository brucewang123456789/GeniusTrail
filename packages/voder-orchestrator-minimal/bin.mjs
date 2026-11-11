#!/usr/bin/env node
import {resolve,join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {readFile} from 'node:fs/promises';
import {runWorkflow} from './lib/engine.mjs';
import {redactError,regular} from './lib/safety.mjs';
import {Journal} from './lib/journal.mjs';
const mode='minimal';
const root=resolve(process.env.VODER_ROOT??fileURLToPath(new URL('../../',import.meta.url)));
const args=process.argv.slice(2);
try{
 if(args.length===1&&args[0]==='--doctor'){
  const [major,minor]=process.versions.node.split('.').map(Number);
  if(major<22||(major===22&&minor<16))throw new Error('NODE_22_16_REQUIRED');
  for(const p of ['packages/voder-providers/dist/daemon_contract.js','packages/voder-providers/bin/daemon-provider-bridge.mjs','packages/voder-sandbox-modal/src/cli.ts','packages/voder-sandbox-modal/src/manifest.ts','packages/voder-cli/dist/core.js'])await regular(join(root,p),2*1024*1024);
  console.log('VODER_ORCHESTRATOR_PREFLIGHT=PASS');console.log('PROFILE='+mode);console.log('NATIVE_VGR_AUTHORITY=NOT_CONNECTED');console.log('REAL_MODAL_API=NOT_TESTED');
 }else if(args.length===2&&args[0]==='--audit'){
  const entries=await new Journal(join(resolve(args[1]),'events.jsonl')).verify();
  console.log('JOURNAL_INTEGRITY=PASS');console.log('EVENTS='+entries.length);console.log('LAST_EVENT='+(entries.at(-1)?.event??'NONE'));
 }else if(args.length===2&&args[0]==='--config'){
  const config=JSON.parse(await readFile(resolve(args[1]),'utf8'));
  const result=await runWorkflow(config,{formal:mode==='formal',voderRoot:root});
  console.log(JSON.stringify(result,null,2));
  // A staged candidate is not authorized VGR delivery; keep nonzero exit.
  process.exitCode=result.status==='VERIFIED_STAGED_VGR_PENDING'?4:3;
 }else{throw new Error('USAGE: node bin.mjs --doctor | --audit /absolute/runDir | --config /absolute/task.json');}
}catch(e){console.error('VODER_ORCHESTRATOR_BLOCKED='+redactError(e));process.exitCode=3;}
