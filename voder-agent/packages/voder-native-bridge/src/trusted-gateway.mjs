import {isAuthorizedSourcePath} from '../../voder-agent-core/src/contracts.mjs';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import { spawn } from 'node:child_process';
import { openSync, writeSync, fsyncSync, closeSync, constants, mkdirSync, lstatSync, realpathSync } from 'node:fs';
import { randomUUID, createHash } from 'node:crypto';
import { access, realpath, stat } from 'node:fs/promises';
import { isAbsolute, join, relative, resolve } from 'node:path';
import { NativeVgrCommitVerifier } from './commit-verifier.mjs';

const IDENT = /^[a-zA-Z0-9._-]{1,64}$/;
const safeIdent = s => IDENT.test(s) && s!=='.' && s!=='..';
const HEX = /^[0-9a-f]{64}$/;
const LIMIT = 300 * 1024;
const STDERR_LIMIT = 4096;
/** Private, create-new diagnostic; cannot confer VGR authority or permit replay. */
function preserveNativeStderr(path, bytes){
  if(!bytes.length)return false;
  let fd;
  try{
    fd=openSync(path,constants.O_WRONLY|constants.O_CREAT|constants.O_EXCL|constants.O_NOFOLLOW,0o600);
    let offset=0;
    while(offset<bytes.length){
      const n=writeSync(fd,bytes,offset,bytes.length-offset);
      if(n<=0)throw Error('private diagnostic write made no progress');
      offset+=n;
    }
    fsyncSync(fd);
    return true;
  }catch{return false}
  finally{if(fd!==undefined)closeSync(fd)}
}


/** Private operator failure artifacts exist independently of the Rust attempt directory.
 * Absence of evidence must NEVER authorize success or re-dispatch.
 * Never persist submitted patch bytes, API secrets, request JSON or raw environment.
 */
function privateFailureRecord(store, input, record, stderr){
  try{
    const st=lstatSync(store);
    if(!st.isDirectory() || st.isSymbolicLink() || realpathSync(store)!==resolve(store))return false;
    const dir=join(store,'gateway-diagnostics');
    try{mkdirSync(dir,{mode:0o700});}catch(e){if(e?.code!=='EEXIST')return false;}
    const dt=lstatSync(dir);
    if(!dt.isDirectory()||dt.isSymbolicLink()||realpathSync(dir)!==resolve(dir))return false;
    if((dt.mode&0o077)!==0)return false;
    const bytes=Buffer.isBuffer(stderr)?stderr:Buffer.alloc(0);
    const redacted=bytes.toString('utf8').slice(0,4096)
      .replace(/(?:sk-|Bearer\s+)[A-Za-z0-9._-]{6,}/gi,'[REDACTED]')
      .replace(/(?:API[_-]?KEY|TOKEN[_-]?SECRET|PASSWORD|AUTHORIZATION)\s*[:=]\s*[^\s,;]+/gi,'[REDACTED]');
    const meta={
      schema:'voder.gateway.operator.failure.v1',
      timestamp:new Date().toISOString(),session_id:input.session_id,
      action_id:input.action_id,run_id:`run-${input.action_id}`,
      stage:record.stage,reason_code:record.reason_code,
      native_status:Number.isInteger(record.status)?record.status:null,
      signal:typeof record.signal==='string'?record.signal:null,
      stdin_error_code:typeof record.stdin_error_code==='string'?record.stdin_error_code:null,
      stderr_sha256:createHash('sha256').update(bytes).digest('hex'),
      stderr_excerpt_redacted:redacted,
      stdout_sha256:typeof record.stdout_sha256==='string'?record.stdout_sha256:null,
      native_attempt_dir_exists:(()=>{try{
        const d=lstatSync(join(store,'attempts',input.action_id));return d.isDirectory()&&!d.isSymbolicLink();
      }catch{return false}})(),
      note:'DIAGNOSTIC_ONLY_NOT_NATIVE_RECEIPT_NOT_COMMIT_AUTHORITY'
    };
    const name=`${input.action_id}.json`;
    // No unsafe action ID can reach here: executeAuthorized validates before dispatch.
    const body=Buffer.from(JSON.stringify(meta,null,2)+'\n');
    if(body.length>16384)return false;
    const fd=openSync(join(dir,name),constants.O_WRONLY|constants.O_CREAT|constants.O_EXCL|constants.O_NOFOLLOW,0o600);
    try{
      let offset=0;
      while(offset<body.length){const n=writeSync(fd,body,offset,body.length-offset);if(n<=0)throw Error('diagnostic stalled');offset+=n;}
      fsyncSync(fd);
    }finally{closeSync(fd)}
    return true;
  }catch{return false}
}

export class GatewayError extends Error {
  constructor(code, message) { super(message); this.name='GatewayError'; this.code=code; }
}
const requiredAbs = (value, name) => {
  if (typeof value!=='string'|| !isAbsolute(value)) throw new GatewayError('CONFIG',`${name} must be absolute`);
  return resolve(value);
};
const validPath = isAuthorizedSourcePath;

/** Policy, executable and attempt store come ONLY from the trusted host constructor. */
export class TrustedVgrGatewayTransport {
  constructor({binary, daemonBinary, policyFile, attemptStore, nodeBinary, modalCli, targetPath,
               timeoutMs=240000, spawnImpl=spawn, targetPaths}={}) {
    this.binary=requiredAbs(binary,'gateway binary');
    this.daemonBinary=requiredAbs(daemonBinary,'daemon binary');
    this.policyFile=requiredAbs(policyFile,'policy file');
    this.attemptStore=requiredAbs(attemptStore,'attempt store');
    this.nodeBinary=requiredAbs(nodeBinary,'Node binary');
    this.modalCli=requiredAbs(modalCli,'Modal driver');
    if(!validPath(targetPath))throw new GatewayError('GATEWAY_TARGET_PATH','Host must pin a canonical source file');
    if(targetPaths!==undefined&&(!Array.isArray(targetPaths)||targetPaths.length<2||targetPaths.length>16||targetPaths.some((p,i)=>!validPath(p)||(i&&targetPaths[i-1]>=p))||targetPaths[0]!==targetPath))throw new GatewayError('CONFIG','Invalid sorted operator batch targets');
    this.targetPath=targetPath;this.targetPaths=targetPaths?Object.freeze([...targetPaths]):undefined;
    if (!Number.isSafeInteger(timeoutMs)||timeoutMs<1000||timeoutMs>900000)throw new GatewayError('CONFIG','invalid deadline');
    this.timeoutMs=timeoutMs;this.spawnImpl=spawnImpl;
    this.attempted=new Set();
  }
  /** Read-only native contract gate, before model calls; no attempt allocation or replay. */
  async preflight(){
    let result;
    try{result=await promisify(execFile)(this.binary,['--check-policy',this.policyFile],{timeout:12000,maxBuffer:32768,env:process.env});}
    catch{throw new GatewayError('GATEWAY_PREFLIGHT','Native capability or pinned policy rejected; no model request permitted');}
    let data;try{data=JSON.parse(result.stdout);}catch{throw new GatewayError('GATEWAY_CAPABILITY','Invalid native capability report');}
    const expected=this.targetPaths??[this.targetPath];
    if(data?.schema!=='voder.gateway.preflight.v1'||data.path_contract!=='voder.source-path.v1'||
       JSON.stringify(data.targets)!==JSON.stringify(expected)||data.status!=='PASS')
      throw new GatewayError('GATEWAY_CAPABILITY','Native/host target contract mismatch');
    return Object.freeze(data);
  }
  attemptDir(actionId) {
    if (!safeIdent(actionId))throw new GatewayError('INVALID_ACTION','Unsafe action identity');
    return join(this.attemptStore,'attempts',actionId);
  }
  async executeAuthorized(input,{signal,workspaceRoot}={}){
    if (!input || input.schema!=='voder.agent.action.v1' || !['patch','patch_batch'].includes(input.kind) ||
        !safeIdent(input.session_id??'') || !safeIdent(input.action_id??'') ||
        !safeIdent(input.idempotency_key??'') ||
        (input.kind==='patch'&&(!validPath(input.relative_path)||!HEX.test(input.old_sha256??'')||typeof input.new_content!=='string'||!Buffer.byteLength(input.new_content))))throw new GatewayError('INVALID_ACTION','Only pinned replacement proposals accepted');
    const patches=input.kind==='patch_batch'?input.patches:[input];
    if(input.kind==='patch_batch'&&(!this.targetPaths||!Array.isArray(patches)||patches.length!==this.targetPaths.length||patches.some((p,i)=>!p||Object.keys(p).sort().join(',')!=='new_content,old_sha256,relative_path'||p.relative_path!==this.targetPaths[i]||!HEX.test(p.old_sha256??'')||typeof p.new_content!=='string'||!Buffer.byteLength(p.new_content))))throw new GatewayError('SCOPE','Batch must match complete exact operator targets');
    if(input.kind==='patch'&&(this.targetPaths||input.relative_path!==this.targetPath))throw new GatewayError('SCOPE','Model target differs from trusted profile');
    if(signal?.aborted)throw new GatewayError('CANCELLED','Pre-dispatch cancelled');
    if (this.attempted.has(input.action_id))throw new GatewayError('DUPLICATE','Attempt already consumed');
    this.attempted.add(input.action_id); // fail-closed even if spawn fails or outcome unknown
    // Rust creates attempt directory atomically; existing directory is an immediate refusal.
    const args=['--policy',this.policyFile,'--attempt-root',this.attemptDir(input.action_id)];
    const payload=JSON.stringify(input.kind==='patch_batch'?{schema:'voder.gateway.request.v2',kind:'patch_batch',session_id:input.session_id,action_id:input.action_id,idempotency_key:input.idempotency_key,patches:input.patches}:{schema:'voder.gateway.request.v1',kind:'patch',session_id:input.session_id,action_id:input.action_id,idempotency_key:input.idempotency_key,relative_path:input.relative_path,old_sha256:input.old_sha256,new_content:input.new_content});
    // No shell; executable/driver paths are pinned by the trusted operator, never model input.
    return new Promise((fulfill,reject)=>{
      let settled=false;let stdout=Buffer.alloc(0);let stderr=Buffer.alloc(0);
      const rejectOnce=e=>{if(settled)return;settled=true;cleanup();reject(e)};
      const fulfillOnce=e=>{if(settled)return;settled=true;cleanup();fulfill(e)};
      let child;
      let timer;
      let stdinFailure=null;
      const record=(stage,reason_code,extra={})=>privateFailureRecord(this.attemptStore,input,{
        stage,reason_code,stdin_error_code:stdinFailure?.code??null,...extra,
        stdout_sha256:createHash('sha256').update(stdout).digest('hex')
      },stderr);
      const abort=()=>{child?.kill('SIGKILL');const saved=record('ABORT','OUTCOME_UNKNOWN');rejectOnce(new GatewayError('OUTCOME_UNKNOWN',`Aborted native action; no automatic retry; operator_diagnostic_saved=${saved}`))};
      const cleanup=()=>{clearTimeout(timer);signal?.removeEventListener?.('abort',abort)};
      try {child=this.spawnImpl(this.binary,args,{stdio:['pipe','pipe','pipe'],shell:false,
        env:{...process.env,VODER_NODE_BINARY:this.nodeBinary,VODER_MODAL_CLI:this.modalCli}})}
      catch(e){const saved=record('SPAWN','NATIVE_SPAWN');rejectOnce(new GatewayError('NATIVE_SPAWN',`Native process could not start; operator_diagnostic_saved=${saved}`));return}
      child.stdout.on('data',c=>{
        stdout=Buffer.concat([stdout,c]);
        if(stdout.length>LIMIT){child.kill('SIGKILL');const saved=record('STDOUT','FRAME_LIMIT');rejectOnce(new GatewayError('FRAME_LIMIT',`Unbounded gateway response; operator_diagnostic_saved=${saved}`))}
      });
      child.stderr.on('data',c=>{if(stderr.length<STDERR_LIMIT)stderr=Buffer.concat([stderr,c.subarray(0,STDERR_LIMIT-stderr.length)])});
      child.on('error',e=>{const saved=record('PROCESS_ERROR','NATIVE_SPAWN');rejectOnce(new GatewayError('NATIVE_SPAWN',`Native process error; operator_diagnostic_saved=${saved}`))});
      child.on('close',(code,term)=>{
        if(settled)return;
        if(code!==0 || term || stdinFailure){
          // Keep native diagnostics local; model-facing RPC must not leak stderr.
          const saved=preserveNativeStderr(join(this.attemptDir(input.action_id),'gateway.stderr'),stderr);
          const recorded=record('PROCESS_CLOSE','NATIVE_REJECTED',{status:code,signal:term});
          return rejectOnce(new GatewayError('NATIVE_REJECTED',
            `VGR gateway rejected action (status=${code}, signal=${term}, stdin_failure=${stdinFailure?.code??'none'}); receipt unavailable; private_stderr_saved=${saved}; operator_diagnostic_saved=${recorded}; operator reconciliation required`));
        }
        try {
          const parsed=JSON.parse(stdout.toString('utf8'));
          if(parsed?.authority!=='VODER_NATIVE_VGR'||parsed?.status!=='COMMITTED'||
             parsed.session_id!==input.session_id || parsed.action_id!==input.action_id ||
             parsed.run_id!==`run-${input.action_id}`||!HEX.test(parsed.commit_digest??'')||
             !HEX.test(parsed.journal_digest??''))
            throw new Error('gateway output cannot establish exact action/commit identity');
          // A plausible JSON payload remains untrusted until separate Rust journal verification.
          fulfillOnce(parsed);
        } catch(e){const saved=record('PARSE_RECEIPT','BAD_RECEIPT',{status:code,signal:term});rejectOnce(new GatewayError('BAD_RECEIPT',`Rust receipt rejected; operator_diagnostic_saved=${saved}`))}
      });
      // timeoutMs is a legacy foreground wait budget, NOT an execution deadline.
      // The Rust VGR action owns durable authorization and commits exactly once.
      // Never kill that action merely because foreground waiting took longer:
      // Modal may still be uploading inputs, scheduling, verifying or sealing a receipt.
      // Explicit AbortSignal cancellation remains fail-closed and non-replayable.
      // Completion is event-driven via the child close/receipt, with no blind retry.
      signal?.addEventListener?.('abort',abort,{once:true});
      // Native process may close its read end before stdin finishes writing.
      // Capture that transport failure; do not turn it into an unhandled crash.
      child.stdin.on?.('error',e=>{stdinFailure=e;});
      try{child.stdin.end(payload)}catch(e){stdinFailure=e;}

    });
  }
}

/** State root derived from a trusted store + host-generated action id, not receipt path. */
export class GatewayCommitVerifier {
  constructor({daemonBinary,attemptStore,timeoutMs=12000,observerFactory}={}){
    this.daemonBinary=requiredAbs(daemonBinary,'daemon');
    this.attemptStore=requiredAbs(attemptStore,'attempt store');
    this.timeoutMs=timeoutMs;this.observerFactory=observerFactory;
  }
  async confirmCommit(receipt,input){
    if(!IDENT.test(input?.action_id??'')||input.action_id!==receipt?.action_id||
       receipt?.run_id!==`run-${input.action_id}`)throw new GatewayError('IDENTITY','Action/run id mismatch');
    const stateDir=join(this.attemptStore,'attempts',input.action_id,'daemon-state');
    const verifier=new NativeVgrCommitVerifier({binary:this.daemonBinary,stateDir,timeoutMs:this.timeoutMs,
      ...(this.observerFactory?{observerFactory:this.observerFactory}: {})});
    return verifier.confirmCommit(receipt,input);
  }
}
