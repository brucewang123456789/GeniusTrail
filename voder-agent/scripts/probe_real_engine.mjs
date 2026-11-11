#!/usr/bin/env node
/** Cloud-only: real upstream session and tool registry; no model billing and no mutation. */
import {mkdtemp,mkdir,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {strict as assert} from 'node:assert';
import {PiModelHost} from '../packages/voder-agent-core/src/pi-host.mjs';
import {repositoryReadTools} from '../packages/voder-agent-core/src/repository-tools.mjs';

const cwd=await mkdtemp(join(tmpdir(),'voder-real-upstream-'));
let session;
try{
  await mkdir(join(cwd,'src'));
  await writeFile(join(cwd,'src','check.txt'),'Voder tool isolation probe\n');
  let requests=0;
  const tools=[{
    name:'voder_authorized_action',description:'Native VGR-only mutation; unconfigured here',
    async execute(){requests++;throw Object.assign(new Error('No native policy'),{code:'DENIED'});}
  },...repositoryReadTools(cwd)];
  session=await new PiModelHost().createSession({cwd,systemPrompt:'Voder audit. No external action or side effects.',tools});
  const inventory=session.inspectTools();
  const expected=['voder_authorized_action','voder_read_file','voder_list_files','voder_search_code'];
  assert.deepEqual(inventory.active,expected);
  assert.deepEqual(inventory.callable,expected);
  assert.deepEqual(inventory.registered,expected);
  assert.equal(requests,0);
  console.log('REAL_PINNED_PI_MULTI_TOOL_SESSION=PASS');
  console.log('REAL_PINNED_PI_EXACT_REGISTERED_ACTIVE_CALLABLE=PASS');
  console.log('REAL_PROVIDER_REQUESTS_ISSUED=0');
  console.log('NATIVE_SIDE_EFFECTS_ISSUED=0');
  console.log('RUST_GATEWAY_MODEL_END_TO_END=NOT_TESTED');
}finally{session?.dispose();await rm(cwd,{recursive:true,force:true});}
