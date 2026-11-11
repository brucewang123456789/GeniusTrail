import {createHash} from 'node:crypto';
import {snapshotWorkspace} from './manifest.ts';
import {inputArchive,uploadArchive,type InputArchive} from './input-archive.ts';
import {ResourceGuard} from './resources.ts';
import {validateTask,validateConfig,type SandboxTask,type Receipt} from './contract.ts';
import type {ModalDriver,RemoteProcess} from './driver.ts';
const sha256=(bytes:string):string=>createHash('sha256').update(bytes).digest('hex');
/** TaskExecStart may be accepted remotely before the client sees UNAVAILABLE.
 * Retrying is allowed ONLY for cryptographically pinned, isolated, side-effect-free
 * verification commands (or /bin/mkdir -p on sandbox-local input directories).
 * Model-proposed arbitrary commands remain single-attempt and fail closed. */
const TASK_EXEC_START_TIMEOUT = /\/modal\.task_command_router\.TaskCommandRouter\/TaskExecStart[\s\S]*?UNAVAILABLE[\s\S]*?(?:no connection established|connection attempt timed out|connection timed out)/i;
// The hard sandbox lifetime must cover transfer, integrity check and the test itself.
// The *host* deadline is longer than the provider lifetime, allowing cleanup/evidence.
// This preparation budget is independently bounded; it is not a limit on candidate count.
export const DEFAULT_SETUP_BUDGET_MS = 720_000;
export const MIN_SETUP_BUDGET_MS = 300_000;
export const MAX_SETUP_BUDGET_MS = 1_800_000;
function setupBudget(task: SandboxTask): number {
  const setup=task.setupBudgetMs ?? DEFAULT_SETUP_BUDGET_MS;
  if(!Number.isSafeInteger(setup)||setup<MIN_SETUP_BUDGET_MS||setup>MAX_SETUP_BUDGET_MS)
    throw new Error('INVALID_SANDBOX_SETUP_BUDGET');
  return setup;
}
const MAX_READONLY_RETRIES = 2;
const RETRY_DELAYS_MS = [600, 1200] as const;
function isTaskExecStartConnectTimeout(error:unknown):boolean {
  return TASK_EXEC_START_TIMEOUT.test(error instanceof Error ? error.message : String(error));
}
function isImmutableReadOnlyTask(program:string,args:string[],inputFiles:ReadonlySet<string>):boolean {
  // Exact accepted command shapes only. No shell, redirection, recursion,
  // network-capable tools, ambient file reads or externally writable paths.
  if ((program==='/usr/bin/grep'||program==='/bin/grep') && args.length===3 &&
      args[0]==='-Fx' && inputFiles.has(args[2] ?? '')) return true;
  if ((program==='/usr/bin/cat'||program==='/bin/cat') && args.length>0 &&
      args.every(a=>inputFiles.has(a))) return true;
  return false;
}
async function execWithSafeRecovery(
  sandbox:Awaited<ReturnType<ModalDriver['create']>>,
  argv:string[],
  opts:{workdir?:string;timeoutMs?:number},
  phase:string,
  safeToRepeat:boolean,
  deadline:number,
):Promise<RemoteProcess> {
  const attempts=safeToRepeat?MAX_READONLY_RETRIES+1:1;
  for(let i=0;i<attempts;i++){
    try {
      const proc=await sandbox.exec(argv,opts);
      if(i>0)process.stderr.write(`VODER_MODAL_RPC_RECOVERED=YES; PHASE=${phase}; ATTEMPTS=${i+1}; DUPLICATE_READONLY_EXECUTION_POSSIBLE=YES\n`);
      return proc;
    }catch(error){
      if(!isTaskExecStartConnectTimeout(error))throw error;
      // Even on client connection failure, the remote command may have run.
      // For a state-changing or unrecognized tool, never retry.
      if(!safeToRepeat){
        throw new Error(`MODAL_RPC_UNCERTAIN_NO_REPLAY; PHASE=${phase}; CLASS=TASK_EXEC_START_CONNECT_TIMEOUT; ATTEMPTS=1; `+
          (error instanceof Error?error.message:String(error)));
      }
      const pause=RETRY_DELAYS_MS[i];
      if(pause===undefined || Date.now()+pause>=deadline){
        throw new Error(`MODAL_READONLY_RETRY_EXHAUSTED; PHASE=${phase}; CLASS=TASK_EXEC_START_CONNECT_TIMEOUT; ATTEMPTS=${i+1}; `+
          (error instanceof Error?error.message:String(error)));
      }
      process.stderr.write(`VODER_MODAL_RPC_RETRY=${i+1}/${MAX_READONLY_RETRIES}; PHASE=${phase}; CLASS=TASK_EXEC_START_CONNECT_TIMEOUT; ONLY_IDEMPOTENT_SANDBOX_ACTION=YES\n`);
      await new Promise<void>(resolve=>setTimeout(resolve,pause));
    }
  }
  throw new Error('MODAL_RETRY_INTERNAL_ERROR');
}

type ByteBudget={remaining:number;limited:boolean};
async function boundedStream(stream:ReadableStream<string|Uint8Array>,budget:ByteBudget,signal?:AbortSignal):Promise<string>{
  // Preserve evidentiary integrity: never silently replace non-UTF8 remote
  // bytes with U+FFFD and claim that the resulting digest is the real output.
  signal?.throwIfAborted();
  const reader=stream.getReader();const cancel=()=>{void reader.cancel(signal?.reason).catch(()=>{});};signal?.addEventListener('abort',cancel,{once:true});const decoder=new TextDecoder('utf-8',{fatal:true,ignoreBOM:true});let text='';
  try{
    for(;;){signal?.throwIfAborted();const {done,value}=await reader.read();signal?.throwIfAborted();if(done)break;
      const bytes=typeof value==='string'?Buffer.from(value,'utf8'):Buffer.from(value);
      const take=Math.min(bytes.length,budget.remaining);
      text+=decoder.decode(bytes.subarray(0,take),{stream:true});
      budget.remaining-=take;
      if(take<bytes.length){budget.limited=true;await reader.cancel().catch(()=>{});break;}
    }
    // A byte cap may cut a multibyte codepoint in half. Dropping only that
    // unfinished codepoint is valid when outputLimited=true; decoding it as
    // replacement text would forge bytes, and throwing would lose the useful
    // truncation receipt.
    if(!budget.limited)text+=decoder.decode();
  }finally{signal?.removeEventListener('abort',cancel);reader.releaseLock();}
  return text;
}
async function captureProcess(p:RemoteProcess,cap:number,signal?:AbortSignal):Promise<{exitCode:number;stdout:string;stderr:string;limited:boolean}>{
  const budget:ByteBudget={remaining:cap,limited:false};
  // Drain stdout and stderr concurrently; the limit applies to *combined* bytes.
  const [stdout,stderr,exitCode]=await Promise.all([
    boundedStream(p.stdout,budget,signal),boundedStream(p.stderr,budget,signal),p.wait(),
  ]);
  if(!Number.isSafeInteger(exitCode)||exitCode<0||exitCode>255)throw new Error('REMOTE_EXIT_CODE_INVALID');
  return {exitCode,stdout,stderr,limited:budget.limited};
}
export async function executeModalTask(task:SandboxTask,driver:ModalDriver,env:NodeJS.ProcessEnv=process.env):Promise<Receipt>{
  validateTask(task);const config=validateConfig(env);
  const guard=new ResourceGuard();let archive:InputArchive|undefined;
  try{
  // No sandbox is created until all missing/unsafe input conditions have been validated.
  const snapshot=await snapshotWorkspace(task.workspace,task.requiredFiles);
  if(!snapshot.files.length)throw new Error('EMPTY_WORKSPACE');
  // Independent sandboxes still receive independent byte verification. Consolidate
  // thousands of filesystem RPCs into one immutable, hash-bound transport artifact.
  archive=snapshot.files.length>=32||snapshot.totalBytes>8*1024*1024?await inputArchive(snapshot,guard):undefined;
  const startedAt=new Date().toISOString();
  let sandbox:Awaited<ReturnType<ModalDriver['create']>>|undefined;
  let inFlightError:unknown;
  let phase='SANDBOX_CREATE';
  let remoteAttempted=false;
  guard.checkMetadata(snapshot.files.length*128);
  const immutableFiles=new Set(snapshot.files.map(f=>f.relative));
  const setupAllowance=setupBudget(task);
  const sandboxLifetimeMs=setupAllowance+task.timeoutMs+90_000;
  const transportDeadlineMs=sandboxLifetimeMs+90_000;
  const initiatedAt=Date.now();
  const deadline=initiatedAt+transportDeadlineMs;
  const phaseStarted=new Map<string,number>();
  const phaseStart=(name:string)=>{phase=name;phaseStarted.set(name,Date.now());
    process.stderr.write(`VODER_MODAL_STAGE=BEGIN; NAME=${name}; ELAPSED_MS=${Date.now()-initiatedAt}\n`);};
  const phaseDone=(name:string)=>{
    const started=phaseStarted.get(name);
    process.stderr.write(`VODER_MODAL_STAGE=END; NAME=${name}; STAGE_MS=${started===undefined?'UNKNOWN':Date.now()-started}; ELAPSED_MS=${Date.now()-initiatedAt}\n`);};
  process.stderr.write(`VODER_MODAL_LIFECYCLE=SANDBOX_MAX_MS=${sandboxLifetimeMs}; IDLE_MAX_MS=${sandboxLifetimeMs}; PREPARE_BUDGET_MS=${setupAllowance}; TASK_MAX_MS=${task.timeoutMs}; HOST_MAX_MS=${transportDeadlineMs}; INPUT_FILES=${snapshot.files.length}; INPUT_BYTES=${snapshot.totalBytes}\n`);

  try{
    phaseStart('SANDBOX_CREATE');
    sandbox=await guard.run<Awaited<ReturnType<ModalDriver['create']>>>(()=>driver.create(config.app,config.image,{
      timeoutMs:sandboxLifetimeMs,idleTimeoutMs:sandboxLifetimeMs,
      memoryMiB:task.memoryMiB,cpu:task.cpu,blockNetwork:true,
    }).then(async owned=>{
      if(guard.controller.signal.aborted){await owned.terminate();throw guard.controller.signal.reason;}
      return {sandboxId:owned.sandboxId,
        filesystem:{writeBytes:(bytes:Uint8Array,path:string)=>guard.run(()=>owned.filesystem.writeBytes(bytes,path)),
          writeText:(text:string,path:string)=>guard.run(()=>owned.filesystem.writeText(text,path))},
        exec:(command:string[],options?:{workdir?:string;timeoutMs?:number})=>guard.run(()=>owned.exec(command,options)),
        terminate:()=>owned.terminate()};
    }));
    process.stderr.write('VODER_OWNED_SANDBOX_ID='+sandbox.sandboxId+'\n');
    phaseDone('SANDBOX_CREATE');
    // Modal does not create parent directories when filesystem.writeBytes() is
    // called. Materialize every parent before uploading any input bytes.
    // Path components have already passed assertRelative() in the snapshot.
    const directories=new Set<string>(['/workspace']);
    for(const file of archive?[]:snapshot.files){
      const parts=file.relative.split('/');
      let path='/workspace';
      for(const part of parts.slice(0,-1)){
        path+=`/${part}`;
        directories.add(path);
      }
    }
    const ordered=[...directories].sort();
    phaseStart('REMOTE_MKDIR');
    for(let i=0;i<ordered.length;i+=64){
      remoteAttempted=true;
      const setup=await execWithSafeRecovery(sandbox,['/bin/mkdir','-p',...ordered.slice(i,i+64)],{
        timeoutMs:Math.min(task.timeoutMs,120_000),
      },phase,true,deadline);
      const result=await guard.run(()=>captureProcess(setup,4096,guard.controller.signal));
      if(result.exitCode!==0||result.limited)throw new Error('REMOTE_DIRECTORY_SETUP_FAILED');
    }
    phaseDone('REMOTE_MKDIR');
    phaseStart('REMOTE_UPLOAD');
    if(archive){
      const parts=await uploadArchive(archive,sandbox.filesystem,guard);
      // Fixed, non-model shell program; upload paths are generated numeric names.
      // State-changing concatenation is never replayed after an ambiguous RPC.
      remoteAttempted=true;
      const combine=await execWithSafeRecovery(sandbox,['/bin/sh','-c',
        'rc=0; for part in /tmp/voder-input.part-*; do cat "$part" || { rc=$?; break; }; done > /tmp/voder-input.tar.gz; test "$rc" -eq 0'],
        {timeoutMs:120_000},'REMOTE_CHUNK_ASSEMBLY',false,deadline);
      const combined=await guard.run(()=>captureProcess(combine,8192,guard.controller.signal));
      if(combined.exitCode!==0||combined.limited)throw Error('REMOTE_ARCHIVE_ASSEMBLY_FAILED');
      await sandbox.filesystem.writeText(`${archive.sha256}  voder-input.tar.gz\n`,'/tmp/voder-archive.sha256');
      process.stderr.write(`VODER_MODAL_INPUT_PACK=FILES=${snapshot.files.length}; SOURCE_BYTES=${snapshot.totalBytes}; ARCHIVE_BYTES=${archive.size}; UPLOAD_PARTS=${parts}; SNAPSHOT_SHA256=${snapshot.digest}; ARCHIVE_SHA256=${archive.sha256}; LOCAL_ARTIFACT_CACHE_HIT=${archive.cacheHit}; VERDICT_REUSE=NO\n`);
      phaseDone('REMOTE_UPLOAD');
      phaseStart('REMOTE_ARCHIVE_VERIFY');remoteAttempted=true;
      const check=await execWithSafeRecovery(sandbox,['/usr/bin/sha256sum','--quiet','-c','voder-archive.sha256'],
        {workdir:'/tmp',timeoutMs:120_000},phase,true,deadline);
      const checked=await guard.run(()=>captureProcess(check,8192,guard.controller.signal));
      if(checked.exitCode!==0||checked.limited)throw new Error('REMOTE_ARCHIVE_INTEGRITY_FAILED');
      phaseDone('REMOTE_ARCHIVE_VERIFY');
      phaseStart('REMOTE_EXTRACT');
      // Extraction is state-changing: an ambiguous RPC is never replayed.
      const extract=await execWithSafeRecovery(sandbox,['/bin/tar','--extract','--gzip','--file','/tmp/voder-input.tar.gz',
        '--directory','/workspace','--no-same-owner','--no-same-permissions'],
        {workdir:'/tmp',timeoutMs:120_000},phase,false,deadline);
      const extracted=await guard.run(()=>captureProcess(extract,8192,guard.controller.signal));
      if(extracted.exitCode!==0||extracted.limited)throw new Error('REMOTE_ARCHIVE_EXTRACTION_FAILED');
      phaseDone('REMOTE_EXTRACT');
    }else{
    guard.check(32*1024*1024);
    for(let i=0;i<snapshot.files.length;i+=8){
      await Promise.all(snapshot.files.slice(i,i+8).map(f=>sandbox!.filesystem.writeBytes(f.bytes,`/workspace/${f.relative}`)));
    }
    await sandbox.filesystem.writeText(snapshot.checksumText,'/workspace/.voder-input.sha256');
    phaseDone('REMOTE_UPLOAD');
    }
    phaseStart('REMOTE_INPUT_VERIFY');
    remoteAttempted=true;
    const verify=await execWithSafeRecovery(sandbox,['/usr/bin/sha256sum','--quiet','-c','.voder-input.sha256'],{
      workdir:'/workspace',timeoutMs:Math.min(task.timeoutMs,120_000),
    },phase,true,deadline);
    const checked=await guard.run(()=>captureProcess(verify,8192,guard.controller.signal));
    if(checked.exitCode!==0||checked.limited)throw new Error('REMOTE_INPUT_INTEGRITY_FAILED');
    phaseDone('REMOTE_INPUT_VERIFY');
    phaseStart('REMOTE_TASK_EXEC');
    remoteAttempted=true;
    let timer:ReturnType<typeof setTimeout>|undefined;
    const deadlineError=new Error('VODER_TASK_WALL_DEADLINE');
    let timeout=false;
    let output:{exitCode:number;stdout:string;stderr:string;limited:boolean};
    const work=(async()=>{
      const proc=await execWithSafeRecovery(sandbox!,[task.program,...task.args],
        {workdir:'/workspace',timeoutMs:task.timeoutMs},phase,
        isImmutableReadOnlyTask(task.program,task.args,immutableFiles),deadline);
      return guard.run(()=>captureProcess(proc,task.maxOutputBytes,guard.controller.signal));
    })();
    try{output=await Promise.race([work,new Promise<never>((_,reject)=>{timer=setTimeout(()=>reject(deadlineError),task.timeoutMs);})]);}
    catch(error){
      const message=error instanceof Error?error.message:String(error);
      if(error!==deadlineError&&!/^Deadline exceeded while streaming stdio for exec [A-Za-z0-9-]+$/.test(message))throw error;
      timeout=true;output={exitCode:124,stdout:'',stderr:'',limited:false};
    }finally{if(timer!==undefined)clearTimeout(timer);}
    phaseDone('REMOTE_TASK_EXEC');
    return {
      protocol:'voder-sandbox-receipt-v1',provider:'modal',runId:task.runId,actionId:task.actionId,
      sandboxId:sandbox.sandboxId,image:config.image,inputDigest:snapshot.digest,
      files:snapshot.files.length,bytes:snapshot.totalBytes,exitCode:output.exitCode,
      timeout,outputLimited:output.limited,...(timeout?{reason:'TASK_DEADLINE_EXCEEDED; SYNTHETIC_EXIT_124; PARTIAL_OUTPUT_OMITTED'}:{}),
      stdoutSha256:sha256(output.stdout),stderrSha256:sha256(output.stderr),
      stdout:output.stdout,stderr:output.stderr,startedAt,endedAt:new Date().toISOString(),status:'EXECUTED',
    };
  }catch(error){
    // Local stage and remote-outcome classification are forensic facts, not
    // evidence that an ambiguous remote RPC was never delivered.
    const msg=error instanceof Error?error.message:String(error);
    // Never interpret a shutdown response as a candidate-test verdict.
    const failureClass=/Sandbox is shutting down|FAILED_PRECONDITION.*shutting down/i.test(msg)
      ? 'SANDBOX_SHUTTING_DOWN' : 'SANDBOX_TRANSPORT_OR_SETUP';
    const detailed=new Error(`MODAL_PHASE=${phase}; MODAL_FAILURE_CLASS=${failureClass}; ELAPSED_MS=${Date.now()-initiatedAt}; SANDBOX_MAX_MS=${sandboxLifetimeMs}; MODAL_OUTCOME=${remoteAttempted?'NOT_PROVEN_STOPPED':'NO_REMOTE_EXEC_RPC_ATTEMPTED'}; ${msg}`);
    inFlightError=detailed;
    throw detailed;
  }
  finally{
    try{if(sandbox){try{phaseStart('SANDBOX_CLEANUP');
        if(guard.controller.signal.aborted){
          let cleanupTimer:ReturnType<typeof setTimeout>|undefined;
          try{await Promise.race([sandbox.terminate(),new Promise<never>((_,reject)=>{cleanupTimer=setTimeout(()=>reject(Error('RESOURCE_STOP_SANDBOX_CLEANUP_UNCONFIRMED')),15000);})]);}
          finally{if(cleanupTimer)clearTimeout(cleanupTimer);}
        }else await sandbox.terminate();
        phaseDone('SANDBOX_CLEANUP');}
      catch(error){if(!inFlightError)throw new Error(`MODAL_TERMINATION_FAILED:${String(error)}`);
        throw new AggregateError([inFlightError,error],`MODAL_RUN_AND_CLEANUP_FAILED: PRIMARY=${String(inFlightError)}; CLEANUP=${String(error)}`);}
    }}finally{/* Resource monitor lives until outer cleanup completes. */}
  }
  }finally{try{await archive?.dispose();}finally{await guard.close();}}
}
