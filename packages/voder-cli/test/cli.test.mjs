import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp, mkdir, cp, writeFile, readFile, stat, symlink} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join, resolve} from 'node:path';
import {spawnSync,spawn} from 'node:child_process';
import {createServer} from 'node:http';
import {fileURLToPath} from 'node:url';
import {dirname} from 'node:path';
import {validateEndpoint,validateState,parseOpts,EMPTY_STATE} from '../dist/core.js';
const HERE=dirname(fileURLToPath(import.meta.url));
const PKG=resolve(HERE,'..');const CLI=join(PKG,'dist/cli.js');
const BUNDLE=resolve(PKG,'../..');
const MOCK_PROVIDER_ROOT=process.env.VODER_CLI_TEST_PROVIDER_ROOT;
async function fixture(){
  const top=await mkdtemp(join(tmpdir(),'voder-cli-qa-'));
  const root=join(top,'voder');const config=join(top,'private');
  await mkdir(join(root,'crates/voderd/src'),{recursive:true});await mkdir(join(root,'packages/voder-providers/bin'),{recursive:true});
  await mkdir(join(root,'packages/voder-cli/vendor'),{recursive:true});
  await writeFile(join(root,'Cargo.toml'),'[workspace]\n');
  await writeFile(join(root,'crates/voderd/src/provider_bridge.rs'),'// integration marker');
  await cp(join(PKG,'vendor'),join(root,'packages/voder-cli/vendor'),{recursive:true});
  await writeFile(join(root,'packages/voder-cli/vendor/MANIFEST.json'),await readFile(join(PKG,'vendor/MANIFEST.json')));
  await cp(join(BUNDLE,'packages/voder-providers/bin/daemon-provider-bridge.mjs'),join(root,'packages/voder-providers/bin/daemon-provider-bridge.mjs'));
  if(MOCK_PROVIDER_ROOT){await mkdir(join(root,'packages/voder-providers/dist'),{recursive:true});await cp(join(MOCK_PROVIDER_ROOT,'packages/voder-providers/dist'),join(root,'packages/voder-providers/dist'),{recursive:true});}
  return {top,root,config};
}
function run(f,args,input){return spawnSync(process.execPath,[CLI,...args],{cwd:f.root,env:{...process.env,VODER_ROOT:f.root,VODER_CONFIG_DIR:f.config},encoding:'utf8',input,timeout:6000});}
test('endpoint rejects plain HTTP non-loopback, injected credentials and query strings',()=>{
  assert.throws(()=>validateEndpoint('http://10.0.0.1:8000/v1',true));
  assert.throws(()=>validateEndpoint('https://user:pass@api.test/v1'));
  assert.throws(()=>validateEndpoint('https://api.test/v1?key=123'));
  assert.equal(validateEndpoint('http://127.0.0.1:8080/v1',true),'http://127.0.0.1:8080/v1');
});
test('state schema rejects unknown names and invalid Open profile',()=>{
  assert.throws(()=>validateState({...EMPTY_STATE,openProfiles:{'../../bad':{}}}));
  assert.throws(()=>validateState({...EMPTY_STATE,active:{kind:'cloud',vendor:'unknown',model:'z',protocol:'chat'}}));
});
test('unknown/duplicate CLI option fails closed',()=>{
  assert.throws(()=>parseOpts(['--protocol','a','--protocol','b']));
  assert.throws(()=>parseOpts(['--unknown']));
});
test('plugin registration is idempotent, soft uninstall prevents active selection',async()=>{
  const f=await fixture();
  assert.equal(run(f,['plugin','install','provider-open']).status,0);
  assert.equal(run(f,['plugin','install','provider-open']).status,0);
  let state=JSON.parse(await readFile(join(f.config,'state.json')));
  assert.equal(state.plugins.open,true);
  assert.equal(run(f,['model','add','veriloop','--runtime','vllm','--endpoint','http://127.0.0.1:8001/v1','--model','VeriLoop-E2','--allow-loopback-http']).status,0);
  assert.equal(run(f,['model','use','open:veriloop']).status,0);
  assert.equal(run(f,['plugin','remove','provider-open']).status,0);
  const p=run(f,['doctor']);assert.notEqual(p.status,0);assert.match(p.stderr,/no active model/);
  state=JSON.parse(await readFile(join(f.config,'state.json')));assert.equal(state.active,null);
  assert.equal(run(f,['plugin','install','provider-open']).status,0);
});
test('cloud credential supplied via stdin is private and never printed',async()=>{
  const f=await fixture();
  const secret='private-qa-secret-not-for-output';
  const install=run(f,['plugin','install','provider-cloud']);assert.equal(install.status,0,install.stderr);
  const login=run(f,['model','login','deepseek','--stdin'],secret+'\n');
  assert.equal(login.status,0,login.stderr);assert.doesNotMatch(login.stdout+login.stderr,/private-qa-secret-not-for-output/);
  const credential=join(f.config,'credentials','deepseek.json');
  assert.equal((await stat(credential)).mode&0o077,0);
  assert.equal(run(f,['model','use','deepseek/deepseek-chat']).status,0);
  assert.equal(run(f,['doctor']).status,0);
});
test('no API key in cloud fails safely; invalid open endpoint denied',async()=>{
  const f=await fixture();
  assert.equal(run(f,['plugin','install','provider-cloud']).status,0);
  const r=run(f,['model','use','openai/gpt-example']);assert.notEqual(r.status,0);assert.match(r.stderr,/missing key/);
  const bad=run(f,['model','add','bad','--runtime','vllm','--endpoint','http://192.168.1.4:9000','--model','m','--allow-loopback-http']);
  assert.notEqual(bad.status,0);
});
test('insecure config file is rejected',async()=>{
  const f=await fixture();await mkdir(f.config,{mode:0o700});await writeFile(join(f.config,'state.json'),'{}',{mode:0o644});
  const r=run(f,['plugin','list']);assert.notEqual(r.status,0);assert.match(r.stderr,/insecure/);
});
test('complete CLI -> Provider bridge -> local fake inference server: real HTTP execution and inert receipt', {timeout:20000}, async(t)=>{
  if(!MOCK_PROVIDER_ROOT){t.skip('provider compiled fixture unavailable');return;}
  const server=createServer((req,res)=>{
    if(req.url!=='/v1/chat/completions'){res.writeHead(404).end();return;}
    let body='';req.on('data',d=>{body+=d});req.on('end',()=>{
      const json=JSON.parse(body);assert.equal(json.model,'VeriLoop-E2');
      res.writeHead(200,{'content-type':'application/json'});
      res.end(JSON.stringify({model:'VeriLoop-E2',choices:[{finish_reason:'stop',message:{role:'assistant',content:'READY'}}],usage:{prompt_tokens:10,completion_tokens:2,total_tokens:12}}));
    });
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  t.after(()=>server.close());
  const f=await fixture();
  // Child processes must not block the parent Node HTTP event loop.
  const invoke=(args)=>new Promise((resolve,reject)=>{const ch=spawn(process.execPath,[CLI,...args],{env:{...process.env,VODER_ROOT:f.root,VODER_CONFIG_DIR:f.config},stdio:['ignore','pipe','pipe']});let stdout='',stderr='';ch.stdout.on('data',d=>stdout+=d);ch.stderr.on('data',d=>stderr+=d);ch.once('error',reject);ch.once('close',code=>resolve({code,stdout,stderr}));});
  assert.equal((await invoke(['plugin','install','provider-open'])).code,0);
  assert.equal((await invoke(['model','add','veriloop','--runtime','vllm','--endpoint',`http://127.0.0.1:${server.address().port}/v1`,'--model','VeriLoop-E2','--allow-loopback-http'])).code,0);
  assert.equal((await invoke(['model','use','open:veriloop'])).code,0);
  const doctor=await invoke(['doctor','--live']);assert.equal(doctor.code,0,doctor.stderr);assert.match(doctor.stdout,/VODER_LIVE_MODEL_REQUEST=PASS/);assert.match(doctor.stdout,/VGR_EXECUTION_AUTHORITY=UNCHANGED/);
});
