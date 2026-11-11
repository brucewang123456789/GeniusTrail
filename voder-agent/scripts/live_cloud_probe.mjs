import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { execFileSync } from 'node:child_process';
import { NativeVgrObserver } from '../packages/voder-native-bridge/src/index.mjs';
const root=resolve(process.argv[2] ?? '/root/autodl-tmp');
const binary=resolve(process.argv[3] ?? join(root,'target/debug/voderd'));
let temporary;let client;
try {
  const pi=await import(join(root,'voder-agent/packages/voder-agent-core/node_modules/@earendil-works/pi-coding-agent/dist/index.js'));
  const names=['createAgentSession','DefaultResourceLoader','SessionManager','getAgentDir','defineTool'];
  if(!names.every(n=>typeof pi[n]==='function'))throw Error('Pi exports missing');
  console.log('PI_V1_1_0_EXPORT_GATE=PASS');
  temporary=await mkdtemp(join(tmpdir(),'voder-agent-native-live-'));
  const work=join(temporary,'workspace');
  await import('node:fs/promises').then(x=>x.mkdir(work));
  execFileSync('/usr/bin/git',['-C',work,'init','-q'],{stdio:'ignore'});
  await writeFile(join(work,'README.md'),'Voder native observer isolated smoke test\n');
  execFileSync('/usr/bin/git',['-C',work,'-c','user.name=Voder Probe','-c','user.email=voder-probe@example.invalid','add','README.md']);
  execFileSync('/usr/bin/git',['-C',work,'-c','user.name=Voder Probe','-c','user.email=voder-probe@example.invalid','commit','-qm','initial test fixture']);
  client=await new NativeVgrObserver({binary,stateDir:join(temporary,'native_state')}).start();
  const result=await client.createSession(work);
  const state=await client.state();
  if(result.session_id!==state.session_id || state.runs.length!==0)throw Error('state or identity mismatch');
  console.log('REAL_VGR_STDIO_SESSION_STATE_GATE=PASS');
  try{await client._request('native.commit',{}, {sessionId:result.session_id});throw Error('unsafe commit route exposed');}
  catch(e){if(e.code!=='METHOD_DENIED')throw e;}
  console.log('CLIENT_COMMIT_DENIAL_GATE=PASS');
  console.log('NATIVE_EXECUTION_AND_COMMIT_GATE=NOT_TESTED');
  console.log('PI_REAL_MODEL_GATE=NOT_TESTED');
  console.log('PRODUCTION_GATE=NOT_PASSED');
} catch(e) {console.error(`BRIDGE_LIVE_GATE=FAIL ${e.code??'ERROR'}: ${e.message}`);process.exitCode=1;}
finally {client?.close();if(temporary)await rm(temporary,{recursive:true,force:true});}
