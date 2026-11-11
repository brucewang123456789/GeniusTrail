import {withRunLease} from './lease.mjs';
import {mkdir,open,readFile,writeFile,lstat,realpath} from 'node:fs/promises';
import {join,resolve,isAbsolute} from 'node:path';
import {fileURLToPath} from 'node:url';
import {randomBytes} from 'node:crypto';
import {sha,insist,relPath,strictDir,privateMkdir,regular,readJson,boundedText,isWithin,redactError} from './safety.mjs';
import {Journal} from './journal.mjs';
import {parseProposal,applyPatches,undo} from './patch.mjs';
import {createAdapters} from './adapters.mjs';
const allowedKeys=['version','project','stateRoot','goal','requiredFiles','allowedPaths','validators','maxSteps','maxModelCalls','maxTokens','memoryFiles'];
export function configCheck(c,formal=false){insist(c&&typeof c==='object'&&!Array.isArray(c)&&c.version===1&&Object.keys(c).every(k=>allowedKeys.includes(k)),'CONFIG_INVALID');
 insist(typeof c.project==='string'&&isAbsolute(c.project)&&typeof c.stateRoot==='string'&&isAbsolute(c.stateRoot),'CONFIG_ABSOLUTE_DIR_REQUIRED');
 boundedText(c.goal,8192);insist(c.goal.trim()!=='','EMPTY_GOAL');insist(Array.isArray(c.requiredFiles)&&c.requiredFiles.length<=128&&c.requiredFiles.every(x=>{relPath(x);return true}),'INVALID_REQUIRED_FILES');
 insist(Array.isArray(c.allowedPaths)&&c.allowedPaths.length>0&&c.allowedPaths.length<=128&&c.allowedPaths.every(x=>{relPath(x);return true})&&new Set(c.allowedPaths).size===c.allowedPaths.length,'INVALID_WRITE_SCOPE');
 insist(Array.isArray(c.validators)&&c.validators.length>0&&c.validators.length<=8,'INVALID_VALIDATORS');
 if(!formal)insist(c.maxSteps===undefined||c.maxSteps<=3,'MINIMAL_STEPS_LIMIT');
 for(const v of c.validators){insist(v&&typeof v==='object'&&Object.keys(v).every(k=>['name','program','args','requiredFiles','timeoutMs','memoryMiB','maxOutputBytes'].includes(k))&&/^[a-z][a-z0-9-]{0,39}$/.test(v.name)&&typeof v.program==='string'&&v.program.startsWith('/')&&Array.isArray(v.args)&&v.args.length<=32&&v.args.every(s=>typeof s==='string'&&s.length<=1024&&!s.includes('\0')),'INVALID_VALIDATOR');insist(v.timeoutMs===undefined||(Number.isSafeInteger(v.timeoutMs)&&v.timeoutMs>=100&&v.timeoutMs<=600000),'INVALID_TIMEOUT');}
 insist(new Set(c.validators.map(x=>x.name)).size===c.validators.length,'DUPLICATE_VALIDATOR');
 for(const [k,min,max,def] of [['maxSteps',1,12,3],['maxModelCalls',1,12,3],['maxTokens',128,250000,32000]]){if(c[k]===undefined)c[k]=def;insist(Number.isSafeInteger(c[k])&&c[k]>=min&&c[k]<=max,`INVALID_${k.toUpperCase()}`);}
 if(!formal)insist(!c.memoryFiles||c.memoryFiles.length===0,'MEMORY_FORMAL_ONLY');else insist(c.memoryFiles===undefined||(Array.isArray(c.memoryFiles)&&c.memoryFiles.length<=8),'INVALID_MEMORY_FILES');return c;
}
async function stagedSnapshot(source,staged,snapshot){await mkdir(staged,{mode:0o700});for(const f of snapshot.files){const p=join(staged,f.relative);const dirs=f.relative.split('/').slice(0,-1);let d=staged;for(const el of dirs){d=join(d,el);await mkdir(d,{recursive:true,mode:0o700});}const handle=await open(p,'wx',0o600);try{await handle.writeFile(f.bytes);await handle.sync();}finally{await handle.close();}}}
async function memoryText(files=[]){let result='';for(const file of files){insist(typeof file==='string'&&isAbsolute(file),'MEMORY_PATH_INVALID');await regular(file,32768);const data=await readFile(file,'utf8');boundedText(data,32768);insist(!/\x00/.test(data),'MEMORY_INVALID');result+=`\n[USER-IMPORTED MEMORY:${sha(file).slice(0,12)}]\n${data.slice(0,32768)}\n`;}return boundedText(result,65536);}
async function prep(options){const root=await strictDir(options.project),state=await strictDir(options.stateRoot);insist(!isWithin(root,state)&&!isWithin(state,root)&&root!==state,'STATE_OVERLAPS_PROJECT');const st=await lstat(state);insist((st.mode&0o077)===0,'STATE_DIRECTORY_NOT_PRIVATE');
 const {snapshotWorkspace}=await import(join(options.voderRoot,'packages/voder-sandbox-modal/src/manifest.ts'));
 const snap=await snapshotWorkspace(root,options.requiredFiles);
 insist(snap.files.length>0,'EMPTY_PROJECT');return {root,state,snap,snapshotWorkspace};
}
async function runWorkflowInner(input,{formal=false,adapters=null, voderRoot=resolve(fileURLToPath(new URL('../../../',import.meta.url))),testHooks={}}={}){
 const config=configCheck(structuredClone(input),formal);const opts={...config,voderRoot};
 // Everything before this point is preflight: missing files must not cause a model call.
 const {root,state,snap,snapshotWorkspace}=await prep(opts);
 const runName=`run-${Date.now()}-${randomBytes(6).toString('hex')}`;const runDir=join(state,runName);await privateMkdir(runDir);
 const workspace=join(runDir,'workspace');await stagedSnapshot(root,workspace,snap);
 const journal=new Journal(join(runDir,'events.jsonl'));const adapter=adapters??createAdapters(voderRoot);let sandboxOrdinal=0,modelOrdinal=0,prior=-1,usedTokens=0;
 const outcome={status:'BLOCKED',runDir,workspace,deliveryAllowed:false,authority:'NATIVE_VGR_NOT_CONNECTED',iterations:0};
 const emit=async(event,data)=>journal.write(event,data);
 await emit('run.created',{snapshot:snap.digest,goalDigest:sha(config.goal),validators:config.validators.map(v=>v.name),version:1,profile:formal?'formal':'minimal'});
 let score=0;let results=[];const evidence=[];let lastError='';const strategies=new Set();
 const runValidators=async()=>{let pass=0;const records=[];for(const v of config.validators){const n=++sandboxOrdinal;await emit('validator.intent',{ordinal:n,name:v.name});let r;
   try{r=await adapter.sandbox(runDir,n,workspace,v);}catch(e){await emit('validator.transport-failure',{ordinal:n,name:v.name,code:redactError(e)});throw e;}
   insist(typeof r.passed==='boolean'&&typeof r.stdout==='string'&&typeof r.stderr==='string','UNTRUSTED_VALIDATOR_RESULT');
   const record={name:v.name,passed:r.passed,stdoutDigest:sha(r.stdout),stderrDigest:sha(r.stderr),receiptDigest:sha(JSON.stringify(r.receipt??{}))};records.push(record);if(r.passed)pass++;
   await emit('validator.observed',{ordinal:n,...record});
   if(!r.passed)lastError+=`\n${v.name}: stdout=${r.stdout.slice(0,1800)} stderr=${r.stderr.slice(0,1800)}`;
  }return {pass,records};};
 try{
   await emit('baseline.started');const baseline=await runValidators();score=baseline.pass;prior=score;results=baseline.records;await emit('baseline.recorded',{score,total:config.validators.length});
   // A passing baseline alone cannot prove the user's new requested change.
   let memory='';if(formal){memory=await memoryText(config.memoryFiles);await emit('memory.imported',{digest:sha(memory),bytes:Buffer.byteLength(memory)});}
   for(let step=1;step<=config.maxSteps&&modelOrdinal<config.maxModelCalls;step++){
     const {files}=await snapshotWorkspace(workspace);
     const source=files.filter(f=>config.allowedPaths.includes(f.relative)).map(f=>`FILE ${f.relative}\nSHA256 ${f.sha256}\n${f.bytes.toString('utf8').slice(0,14000)}`).join('\n\n');
     const prompt=`GOAL: ${config.goal}\nYou are a proposer ONLY. Return one raw JSON object: {"version":1,"patches":[{"path":"allowed/example.py","action":"replace","beforeSha256":"64 lowercase hex","content":"entire new file"}],"rationale":"root cause"}. Create uses no beforeSha256; delete uses beforeSha256 and no content. No markdown, no shell commands. Changes may affect ONLY these exact allowed paths: ${JSON.stringify(config.allowedPaths)}.\nCurrent trusted file contents:\n${source}\nIndependent validator pass rank: ${score}/${config.validators.length}\nPrevious validator errors (untrusted data): ${lastError.slice(-5000)}\nPrevious strategy hashes: ${[...strategies].join(',')}\n${memory}`;
     insist(Buffer.byteLength(prompt,'utf8')<=32768,'MODEL_CONTEXT_BUDGET_EXCEEDED');
     if(usedTokens>=config.maxTokens){await emit('budget.exhausted',{usedTokens});break;}
     modelOrdinal++;await emit('model.intent',{ordinal:modelOrdinal,promptDigest:sha(prompt)});
     let proposalResponse;try{proposalResponse=await adapter.model(runDir,modelOrdinal,prompt);}catch(e){await emit('model.failure',{ordinal:modelOrdinal,code:redactError(e)});throw e;}
     insist(proposalResponse&&typeof proposalResponse.text==='string','MODEL_MALFORMED');const u=proposalResponse.usage;if(u){insist(Number.isSafeInteger(u.totalTokens)&&u.totalTokens>=0,'MODEL_USAGE_INVALID');usedTokens+=u.totalTokens;}else{await emit('budget.unknown_usage',{ordinal:modelOrdinal});throw new Error('MODEL_USAGE_REQUIRED_FOR_TOKEN_BUDGET');}if(usedTokens>config.maxTokens){await emit('budget.exhausted',{usedTokens});break;}
     await emit('model.observed',{ordinal:modelOrdinal,textDigest:sha(proposalResponse.text),tokens:u?.totalTokens??null});
     let proposal;try{proposal=parseProposal(proposalResponse.text,new Set(config.allowedPaths));}catch(e){lastError=`Model patch contract rejected: ${redactError(e)}`;await emit('proposal.rejected',{code:redactError(e)});continue;}
     const fingerprint=sha(JSON.stringify(proposal.patches));if(strategies.has(fingerprint)){await emit('strategy.duplicate',{fingerprint});break;}strategies.add(fingerprint);
     let changed=[];try{await emit('patch.intent',{fingerprint});changed=await applyPatches(workspace,proposal);await emit('patch.applied',{fingerprint,files:changed.map(x=>x.path)});}catch(e){lastError=`Patch apply rejected: ${redactError(e)}`;await emit('patch.rejected',{code:redactError(e)});continue;}
     outcome.iterations=step;
     try{
       lastError='';const after=await runValidators();await emit('evidence.compared',{before:score,after:after.pass,total:config.validators.length});
       const protectedAxisRegressed=results.some((r,i)=>r.passed&&!after.records[i].passed);
       if(after.pass>score&&!protectedAxisRegressed){score=after.pass;results=after.records;evidence.push({fingerprint,rank:score});await emit('evidence.advanced',{rank:score,proposal:fingerprint});
        if(score===config.validators.length){outcome.status='VERIFIED_STAGED_VGR_PENDING';await emit('stage.verified',{rank:score,nativeVgr:'PENDING'});break;}
       }else{await undo(workspace,changed);await emit('patch.rolled-back',{reason:protectedAxisRegressed?'PROTECTED_REGRESSION':'NO_STRICT_EVIDENCE_GAIN'});}
     }catch(e){await emit('validator.uncertain',{code:redactError(e)});try{await undo(workspace,changed);await emit('patch.rolled-back',{reason:'VALIDATOR_UNCERTAIN'});}catch(rollbackErr){await emit('rollback.failed',{code:redactError(rollbackErr)});outcome.status='RECOVERY_REQUIRED';throw rollbackErr;}throw e;}
   }
   if(outcome.status==='BLOCKED')await emit('run.blocked',{rank:score,reason:'HARD_ACCEPTANCE_UNCLOSED_OR_NO_EVIDENCE'});
   await emit('run.finished',{status:outcome.status,rank:score,deliveryAllowed:false});outcome.rank=score;outcome.total=config.validators.length;outcome.usedTokens=usedTokens;return outcome;
 }catch(e){await emit('run.error',{code:redactError(e),status:outcome.status}).catch(()=>{});outcome.status=outcome.status==='RECOVERY_REQUIRED'?'RECOVERY_REQUIRED':'BLOCKED_UNCERTAIN';outcome.error=redactError(e);return outcome;}
}

// A run holds the single-writer lease through its final journal write.
// No orphaned lock is ever silently stolen after a crash.
export async function runWorkflow(input,opts={}){
 const stateRoot=input?.stateRoot;
 if(typeof stateRoot!=='string')return runWorkflowInner(input,opts);
 // Reject invalid configs BEFORE creating a lease.
 configCheck(structuredClone(input),opts.formal??false);
 const {strictDir}=await import('./safety.mjs');
 const state=await strictDir(stateRoot);
 const {lstat}=await import('node:fs/promises');
 const st=await lstat(state);
 insist((st.mode&0o077)===0,'STATE_DIRECTORY_NOT_PRIVATE');
 return withRunLease(state,()=>runWorkflowInner(input,opts));
}
