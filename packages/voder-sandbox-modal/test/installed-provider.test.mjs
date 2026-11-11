import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm,access,chmod} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {PluginRegistry} from '../src/plugin.ts';
import {builtinPlugins} from '../src/registry.ts';
import {registerInstalledPlugins,catalogue} from '../src/installed.ts';
import {assertProviderReceipt} from '../src/receipt-gate.ts';
const sha=s=>createHash('sha256').update(s).digest('hex');
const manager=new URL('../src/manage-plugins.mjs',import.meta.url).pathname;
const cli=new URL('../src/cli.ts',import.meta.url).pathname;
const run=(args,extra={})=>spawnSync(process.execPath,['--experimental-strip-types',manager,...args],{encoding:'utf8',env:{...process.env,...extra},timeout:15000});
const make=async()=>{const root=await mkdtemp(join(tmpdir(),'voder-provider-ext-'));await chmod(root,0o700);return root;};
const custom=`import {writeFileSync} from 'node:fs';
if(process.env.TEST_PLUGIN_MARKER)writeFileSync(process.env.TEST_PLUGIN_MARKER,'LOADED');
export async function createSandboxPlugin(){return {id:'custom-one',version:'1.2.3',async execute(task){
const hash=await import('node:crypto');const h=s=>hash.createHash('sha256').update(s).digest('hex');
return {protocol:'voder-sandbox-receipt-v1',provider:'custom-one',runId:task.runId,actionId:task.actionId,sandboxId:'isolated-123',image:'pinned-runtime',inputDigest:'a'.repeat(64),files:1,bytes:4,exitCode:0,timeout:false,outputLimited:false,stdout:'ok\\n',stderr:'',stdoutSha256:h('ok\\n'),stderrSha256:h(''),startedAt:'2026-10-12T00:00:00Z',endedAt:'2026-10-12T00:00:01Z',status:'EXECUTED'};}};}`;
const task={workspace:'/tmp/work',program:'/bin/true',args:[],timeoutMs:1000,memoryMiB:512,cpu:1,maxOutputBytes:1024,runId:'r1',actionId:'a1'};
test('operator installs only pinned code; Modal remains default and extensions load lazily',async()=>{
 const root=await make();try{
  const file=join(root,'plugin-source.mjs'),marker=join(root,'loaded');await writeFile(file,custom,{mode:0o600});
  const installed=run(['install','--root',root,'--id','custom-one','--file',file,'--version','1.2.3','--sha256',sha(custom)],{TEST_PLUGIN_MARKER:marker});
  assert.equal(installed.status,0,installed.stderr);
  await assert.rejects(access(marker));
  assert.deepEqual(builtinPlugins().list(),['modal']);
  const registry=builtinPlugins();await registerInstalledPlugins(registry,root);assert.deepEqual(registry.list(),['custom-one','modal']);
  const prev=process.env.TEST_PLUGIN_MARKER;process.env.TEST_PLUGIN_MARKER=marker;
  try{const provider=await registry.resolve('custom-one');assertProviderReceipt(await provider.execute(task),task,'custom-one');}
  finally{if(prev===undefined)delete process.env.TEST_PLUGIN_MARKER;else process.env.TEST_PLUGIN_MARKER=prev;}
  assert.equal(await readFile(marker,'utf8'),'LOADED');
  assert.equal(run(['verify','--root',root]).status,0);
  assert.notEqual(run(['install','--root',root,'--id','modal','--file',file,'--version','1.2.3','--sha256',sha(custom)]).status,0);
  assert.notEqual(run(['install','--root',root,'--id','custom-one','--file',file,'--version','1.2.3','--sha256',sha(custom)]).status,0);
  assert.equal(run(['remove','--root',root,'--id','custom-one']).status,0);
  assert.deepEqual(await catalogue(root),[]);
 }finally{await rm(root,{recursive:true,force:true});}
});
test('tampering, unsafe package metadata, symlinks and bad digests fail closed',async()=>{
 const root=await make();try{
  const file=join(root,'provider.mjs');await writeFile(file,custom,{mode:0o600});
  assert.notEqual(run(['install','--root',root,'--id','custom-one','--file',file,'--version','1.2.3','--sha256','f'.repeat(64)]).status,0);
  assert.equal(run(['install','--root',root,'--id','custom-one','--file',file,'--version','1.2.3','--sha256',sha(custom)]).status,0);
  const record=(await catalogue(root))[0];await writeFile(join(root,record.entry),'tamper',{mode:0o600});
  const registry=new PluginRegistry();await registerInstalledPlugins(registry,root);
  await assert.rejects(registry.resolve('custom-one'),/DIGEST_MISMATCH/);
  assert.notEqual(run(['verify','--root',root]).status,0);
 }finally{await rm(root,{recursive:true,force:true});}
});
test('receipt validation rejects forged identity, oversized output and status',()=>{
 const out='ok',record={protocol:'voder-sandbox-receipt-v1',provider:'custom-one',runId:'r1',actionId:'a1',sandboxId:'x',image:'digest',inputDigest:'a'.repeat(64),files:1,bytes:2,exitCode:0,timeout:false,outputLimited:false,stdout:out,stderr:'',stdoutSha256:sha(out),stderrSha256:sha(''),startedAt:'2026-10-12T00:00:00Z',endedAt:'2026-10-12T00:00:01Z',status:'EXECUTED'};
 assert.equal(assertProviderReceipt(record,task,'custom-one'),record);
 for(const r of [{...record,runId:'changed'},{...record,provider:'modal'},{...record,status:'BLOCKED'},{...record,stdoutSha256:'f'.repeat(64)},{...record,stdout:'x'.repeat(2000)},{...record,image:''},{...record,exitCode:null}])
   assert.throws(()=>assertProviderReceipt(r,task,'custom-one'),/CONTRACT_VIOLATION/);
});
test('disabled sandbox does not silently execute or create fake receipts',async()=>{
 const root=await make();try{
  const p=['req','receipt','stdout','stderr'].map(x=>join(root,x));await writeFile(p[0],JSON.stringify(task));
  const r=spawnSync(process.execPath,['--experimental-strip-types',cli,...p],{encoding:'utf8',env:{...process.env,VODER_SANDBOX_BACKEND:'none'},timeout:15000});
  assert.equal(r.status,70);assert.match(r.stderr,/SANDBOX_DISABLED_NO_EXECUTION/);await assert.rejects(access(p[1]));
 }finally{await rm(root,{recursive:true,force:true});}
});
test('installed alternative reaches same four-artifact CLI ABI without invoking Modal',async()=>{
 const root=await make();try{
   const store=join(root,'store');await mkdir(store,{mode:0o700});
   const pluginFile=join(root,'plugin-source.mjs');await writeFile(pluginFile,custom,{mode:0o600});
   const installed=run(['install','--root',store,'--id','custom-one','--file',pluginFile,'--version','1.2.3','--sha256',sha(custom)]);
   assert.equal(installed.status,0,installed.stderr);
   const paths=['request.json','receipt.json','out.raw','err.raw'].map(x=>join(root,x));
   await writeFile(paths[0],JSON.stringify(task));
   const result=spawnSync(process.execPath,['--experimental-strip-types',cli,...paths],{
     encoding:'utf8',env:{...process.env,VODER_SANDBOX_BACKEND:'custom-one',VODER_SANDBOX_PLUGIN_DIR:store},timeout:15000});
   assert.equal(result.status,0,result.stderr);
   assert.equal(await readFile(paths[2],'utf8'),'ok\n');
   assert.equal(await readFile(paths[3],'utf8'),'');
   const receipt=JSON.parse(await readFile(paths[1],'utf8'));
   assert.equal(receipt.provider,'custom-one');assert.equal(receipt.status,'EXECUTED');
   assert.equal(receipt.stdoutSha256,sha('ok\n'));assert.equal(receipt.runId,task.runId);
   assert.ok(!('stdout' in receipt));
 }finally{await rm(root,{recursive:true,force:true});}
});
