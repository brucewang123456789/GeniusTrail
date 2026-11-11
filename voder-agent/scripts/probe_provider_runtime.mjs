/** Non-billable live installed Pi SDK provider registration / real tool safety probe. */
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
const root=resolve(process.argv[2]||'/root/autodl-tmp/voder-agent');
try{
 const runtime=await import(pathToFileURL(join(root,'packages/voder-agent-core/src/model-plugin-runtime.mjs')));
 const selected=await runtime.readInstalledMode();
 if(!selected)throw Error('MODEL_PROVIDER_PLUGIN_NOT_INSTALLED');
 // Synthetic key, no network model request; DOES NOT use or reveal operator key.
 process.env.VODER_MODEL_ID=selected==='api'?'deepseek-chat':'voder-local-test-model';
 process.env.VODER_MODEL_API_KEY='VODER_TEST_KEY_NEVER_REAL';
 delete process.env.VODER_MODEL_ENDPOINT;
 delete process.env.VODER_MODEL_VENDOR;
 const hostModule=await import(pathToFileURL(join(root,'packages/voder-agent-core/src/pi-host.mjs')));
 const base=await mkdtemp(join(tmpdir(),'voder-provider-synthetic-'));
 let s;
 try{
  const host=new hostModule.PiModelHost();
  s=await host.createSession({cwd:base,systemPrompt:'No tool execution. Model API probe only.',tools:[{name:'voder_authorized_action',description:'Denied native action',execute:()=>{throw Error('DENIED');}}]});
  const list=s.inspectTools();
  if(list.active.length!==1||list.active[0]!=='voder_authorized_action'||list.callable.length!==1||list.registered.length!==1)throw Error('TOOL_BOUNDARY_VIOLATION');
  console.log('ACTUAL_PI_MODEL_RUNTIME_REGISTRATION=PASS');
  console.log('ACTUAL_PI_TOOL_BOUNDARY=PASS');
  console.log('ACTIVE_PLUGIN_MODE='+selected);
  console.log('REAL_MODEL_API_REQUESTS=0');
 }finally{s?.dispose();await rm(base,{recursive:true,force:true});}
}catch(e){console.error('ACTUAL_PI_MODEL_RUNTIME_REGISTRATION=FAIL');console.error(String(e?.message??e).slice(0,300));process.exitCode=1;}
