'use strict';
/** Fixed Python unittest adapter. Selected source executes only inside the existing pinned Modal backend. */
const fs=require('node:fs/promises'),path=require('node:path'),crypto=require('node:crypto'),{spawn}=require('node:child_process');
const {readScope,hash}=require('./workspace-tools.cjs');
const {nativeEnv}=require('./native-env.cjs');
const {IMAGE,pinnedManagedNode,cacheManagedNode}=require('./native-profile.cjs');
async function executeDriver(node,args,cwd,env){
 return new Promise((resolve,reject)=>{const child=spawn(node,args,{cwd,env,shell:false,stdio:['ignore','pipe','pipe']});let output='',reason=null;const timer=setTimeout(()=>{reason='SANDBOX_TRANSPORT_TIMEOUT';child.kill('SIGKILL');},180000);for(const stream of [child.stdout,child.stderr])stream.on('data',b=>{if(Buffer.byteLength(output)+b.length>32768){reason='SANDBOX_TRANSPORT_OUTPUT_LIMIT';child.kill('SIGKILL');}else output+=b.toString();});child.on('error',e=>{clearTimeout(timer);reject(e);});child.on('close',code=>{clearTimeout(timer);if(reason||code!==0)reject(Error(reason||'SANDBOX_TRANSPORT_FAILED'));else resolve();});});
}
async function runPythonChecks({root,files,agentRoot,storageRoot,execute=executeDriver}){
 if(!Array.isArray(files)||files.some(f=>!f.endsWith('.py'))||!files.some(f=>/(^|\/)test[^/]*\.py$/.test(f)))throw Error('PYTHON_TEST_SCOPE: choose Python source and at least one test*.py');
 const before=await readScope(root,files),env=nativeEnv(process.env),host=path.resolve(agentRoot,'..'),node=await cacheManagedNode(await pinnedManagedNode(host,{details:true}),storageRoot),driver=path.join(host,'packages/voder-sandbox-modal/src/cli.ts');
 const st=await fs.lstat(driver);if(!st.isFile()||st.isSymbolicLink()||await fs.realpath(driver)!==driver)throw Error('SANDBOX_DRIVER_UNSAFE');
 await fs.mkdir(storageRoot,{recursive:true,mode:0o700});const dir=await fs.mkdtemp(path.join(storageRoot,'python-check-')),repo=path.join(dir,'snapshot');await fs.mkdir(repo,{mode:0o700});
 for(const row of before){const dest=path.join(repo,row.relative);await fs.mkdir(path.dirname(dest),{recursive:true,mode:0o700});await fs.writeFile(dest,row.bytes,{flag:'wx',mode:0o600});}
 const request={workspace:repo,program:'/usr/bin/env',args:['/usr/local/bin/python3.10','-I','-B','-m','unittest','discover','-s','.','-p','test*.py'],timeoutMs:30000,memoryMiB:512,cpu:1,maxOutputBytes:262144,runId:'tools-check-'+crypto.randomUUID(),actionId:'readonly-check-'+crypto.randomUUID()};
 const names=['request.json','receipt.json','stdout.raw','stderr.raw'].map(n=>path.join(dir,n));await fs.writeFile(names[0],JSON.stringify(request),{flag:'wx',mode:0o600});
 await execute(node,['--experimental-strip-types',driver,...names],dir,env);
 const read=async(f,max)=>{const s=await fs.lstat(f);if(!s.isFile()||s.isSymbolicLink()||s.size>max)throw Error('SANDBOX_EVIDENCE_UNSAFE');return fs.readFile(f);};
 const receipt=JSON.parse(await read(names[1],1048576)),stdout=await read(names[2],262144),stderr=await read(names[3],262144);
 if(receipt.protocol!=='voder-sandbox-receipt-v1'||receipt.provider!=='modal'||receipt.status!=='EXECUTED'||receipt.runId!==request.runId||receipt.actionId!==request.actionId||receipt.image!==IMAGE||receipt.stdoutSha256!==hash(stdout)||receipt.stderrSha256!==hash(stderr)||!Number.isInteger(receipt.exitCode)||receipt.exitCode<0||receipt.exitCode>255||typeof receipt.timeout!=='boolean'||typeof receipt.outputLimited!=='boolean')throw Error('SANDBOX_RECEIPT_MISMATCH');
 const after=await readScope(root,files);if(after.some((x,i)=>x.sha256!==before[i].sha256))throw Error('SANDBOX_ORIGINAL_SOURCE_CHANGED');
 const diagnostic=Buffer.concat([stdout,stderr]).toString('utf8'),count=diagnostic.match(/Ran (\d+) tests?/),tests=count?Number(count[1]):null;
 return {schema:'voder.tools.sandbox-check.v1',authority:'SANDBOX_TEST_OBSERVATION_NOT_VGR_COMMIT',status:receipt.timeout||receipt.outputLimited?'UNAVAILABLE':receipt.exitCode===0&&tests>0?'PASS':'FAIL',tests_observed:tests,files:before.map(r=>({path:r.relative,sha256:r.sha256})),image:receipt.image,run_id:receipt.runId,diagnostic:diagnostic.slice(-6000),evidence_directory:dir,source_writes:0,native_actions:0};
}
module.exports={runPythonChecks};
