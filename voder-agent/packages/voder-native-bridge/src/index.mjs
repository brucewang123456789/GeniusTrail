import { spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { resolve, isAbsolute } from 'node:path';

/** An untrusted intent and state-observation client, NEVER a VGR commit authority. */
export class NativeProtocolError extends Error {
  constructor(code, message) { super(message); this.name = 'NativeProtocolError'; this.code = code; }
}
const MAX_FRAME = 1024 * 1024;
const READ_METHODS = new Set(['session.resume','state.get','run.replay']);
const METHODS = new Set(['initialize','session.create', ...READ_METHODS]);
const WIRE_ID = /^[a-zA-Z0-9._:-]{1,128}$/;
function validFrame(f, expected) {
  if (!f || f.protocol_version !== 1 || !['response','error'].includes(f.frame_type) ||
      f.request_id !== expected || typeof f.kind !== 'string' || !f.payload ||
      typeof f.payload !== 'object' || Array.isArray(f.payload)) {
    throw new NativeProtocolError('WIRE_MISMATCH','Daemon returned invalid or mismatched frame');
  }
  return f;
}
/**
 * One child process, one ordered request at a time; no autoretry after timeout/exit.
 * This class deliberately has NO executeAuthorized/prepare/commit method.
 */
export class NativeVgrObserver {
  constructor({ binary, stateDir, timeoutMs = 12000, env = process.env }={}) {
    if (!isAbsolute(binary ?? '') || !isAbsolute(stateDir ?? ''))
      throw new NativeProtocolError('INVALID_PATH','Absolute daemon and state paths required');
    if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 100 || timeoutMs > 120000)
      throw new NativeProtocolError('INVALID_TIMEOUT','Timeout outside bounds');
    this.binary = resolve(binary); this.stateDir = resolve(stateDir);
    this.timeoutMs = timeoutMs; this.env = env; this.child = null;
    this.buffer = Buffer.alloc(0); this.pending = null; this.queue = Promise.resolve();
    this.closed = false; this.fault = null; this.sessionId = null; this.initialized = false;
    this.childExit = null; this.shutdownChild = null; this.closeWait = null;
  }
  async start() {
    if (this.child || this.closed) throw new NativeProtocolError('LIFECYCLE','Daemon already started or closed');
    const child = spawn(this.binary, ['--state-dir',this.stateDir],
       {stdio:['pipe','pipe','pipe'],env:this.env});
    this.child = child;
    // The 'close' event means OS pipes and the exclusive daemon lock are released.
    // 'exit' alone is not sufficient; a SIGTERM send is not a shutdown barrier.
    this.childExit = new Promise(resolve=>child.once('close',(code,signal)=>resolve({code,signal})));
    child.stdin.on('error', e=>{if(!this.closed)this._fault('WRITE_ERROR',e.message);});
    child.stdout.on('data', data=>this._receive(data));
    // Never write daemon stderr into protocol or user-facing logs (potential secrets).
    let stderrBytes = 0;
    child.stderr.on('data', data=>{ stderrBytes += data.length; });
    child.on('error',e=>this._fault('SPAWN_FAILED',e.message));
    child.on('exit',(code,signal)=>this._fault('DAEMON_EXIT',`code=${code} signal=${signal}`));
    try {
      const hello=await this._request('initialize',
       {minimum_version:1,maximum_version:1,client_name:'voder-native-observer',requested_features:['jsonl']});
      if (hello.protocol_version !== 1) throw new NativeProtocolError('NEGOTIATION','Unsupported protocol');
      this.initialized = true; return this;
    } catch(e) { await this.closeAndWait().catch(()=>{}); throw e; }
  }
  _fault(code,message) {
    if (this.fault) return;
    this.fault = new NativeProtocolError(code,message);
    if (this.pending) { this.pending.reject(this.fault); this.pending=null; }
  }
  _receive(chunk) {
    if (this.fault || this.closed) return;
    this.buffer=Buffer.concat([this.buffer,chunk]);
    if (this.buffer.length > MAX_FRAME * 2) return this._fault('FRAME_TOO_LARGE','Unbounded stdout');
    while(true) {
      const n=this.buffer.indexOf(10); if(n<0) break;
      if(n+1 > MAX_FRAME) return this._fault('FRAME_TOO_LARGE','Oversized frame');
      const raw=this.buffer.subarray(0,n+1); this.buffer=this.buffer.subarray(n+1);
      const job=this.pending;
      if(!job) return this._fault('UNSOLICITED_FRAME','Unexpected frame from daemon');
      this.pending=null;
      try {
        const frame=validFrame(JSON.parse(raw.toString('utf8')),job.id);
        if(frame.kind!==job.kind) throw new NativeProtocolError('KIND_MISMATCH','Unexpected response method');
        if(frame.frame_type==='error') throw new NativeProtocolError(frame.payload.code ?? 'NATIVE_ERROR',frame.payload.message ?? 'Native rejected');
        job.resolve(frame.payload);
      } catch(e) { job.reject(e); }
    }
    if(this.buffer.length>MAX_FRAME) this._fault('FRAME_TOO_LARGE','Unterminated oversized frame');
  }
  async _request(kind,payload={},options={}) {
    if(!METHODS.has(kind)) throw new NativeProtocolError('METHOD_DENIED','Only observed native methods allowed');
    if (this.closed || this.fault) throw this.fault ?? new NativeProtocolError('CLOSED','Transport closed');
    const work=()=>new Promise((ok,fail)=>{
      if(this.closed || this.fault) return fail(this.fault ?? new NativeProtocolError('CLOSED','Transport closed'));
      const id=`voder-${randomUUID()}`;
      const msg={protocol_version:1,frame_type:'request',request_id:id,kind,payload};
      if(options.sessionId) {
        if(!WIRE_ID.test(options.sessionId))return fail(new NativeProtocolError('INVALID_SESSION','Bad session id'));
        msg.session_id=options.sessionId;
      }
      if(options.idempotencyKey) {
        if(!WIRE_ID.test(options.idempotencyKey))return fail(new NativeProtocolError('INVALID_IDEMPOTENCY','Bad key'));
        msg.idempotency_key=options.idempotencyKey;
      }
      const frame=Buffer.from(JSON.stringify(msg)+'\n');
      if(frame.length > MAX_FRAME) return fail(new NativeProtocolError('FRAME_TOO_LARGE','Request exceeds limit'));
      const timer=setTimeout(()=>{
        this._fault('TIMEOUT','Native response unknown; no retry');
        this.close();
      },this.timeoutMs);
      const cleanup=fn=>value=>{clearTimeout(timer);fn(value);};
      this.pending={id,kind,resolve:cleanup(ok),reject:cleanup(fail)};
      try { this.child.stdin.write(frame,e=>{if(e && !this.closed)this._fault('WRITE_ERROR',e.message)}); }
      catch(e) { this._fault('WRITE_ERROR',e.message); }
    });
    const result=this.queue.then(work);this.queue=result.catch(()=>{});return result;
  }
  async createSession(projectRoot) {
    if(!this.initialized || this.sessionId)throw new NativeProtocolError('LIFECYCLE','Observer already bound or not initialized');
    if(!isAbsolute(projectRoot))throw new NativeProtocolError('INVALID_PATH','Repository root must be absolute');
    const answer=await this._request('session.create',{project_root:resolve(projectRoot)},
       {idempotencyKey:`voder-${randomUUID()}`});
    if(!WIRE_ID.test(answer.session_id ?? '') || answer.status!=='READY')throw new NativeProtocolError('UNVERIFIED_SESSION','Server did not create session');
    this.sessionId=answer.session_id;return answer;
  }
  async resume(sessionId) {
    if(!this.initialized || this.sessionId)throw new NativeProtocolError('LIFECYCLE','Already bound or not initialized');
    const answer=await this._request('session.resume',{}, {sessionId});
    if(answer.session_id!==sessionId || answer.status!=='READY')throw new NativeProtocolError('UNVERIFIED_SESSION','Session not found');
    this.sessionId=sessionId;return answer;
  }
  async state(runId) {
    if(!this.sessionId)throw new NativeProtocolError('LIFECYCLE','Session not bound');
    const body=runId===undefined?{}:{run_id:runId};
    const answer=await this._request('state.get',body,{sessionId:this.sessionId});
    if(answer.session_id!==this.sessionId || !Array.isArray(answer.runs))throw new NativeProtocolError('UNVERIFIED_STATE','State response malformed');
    return answer;
  }
  async replay(runId,afterSequence=0,limit=128) {
    if(!this.sessionId)throw new NativeProtocolError('LIFECYCLE','Session not bound');
    if(!WIRE_ID.test(runId) || !Number.isSafeInteger(afterSequence) || afterSequence<0 || !Number.isSafeInteger(limit) || limit<1 || limit>1024)throw new NativeProtocolError('INVALID_ARGUMENT','Invalid replay bound');
    return this._request('run.replay',{run_id:runId,after_sequence:afterSequence,limit},{sessionId:this.sessionId});
  }
  // No raw tool, commit, native prep or execute route here.
  close(){
    if(this.closed)return;
    this.closed=true;
    const child=this.child;
    this.child=null;
    this.shutdownChild=child;
    if(this.pending){this.pending.reject(new NativeProtocolError('CLOSED','Transport closed'));this.pending=null;}
    if(child && child.exitCode===null && child.signalCode===null)child.kill('SIGTERM');
  }
  /** Await actual child 'close', not just SIGTERM delivery, before opening same state-dir again. */
  async closeAndWait({graceMs=2000,killMs=2000}={}){
    if(!Number.isSafeInteger(graceMs)||graceMs<100||graceMs>120000||
       !Number.isSafeInteger(killMs)||killMs<100||killMs>120000)
      throw new NativeProtocolError('INVALID_TIMEOUT','Invalid shutdown deadline');
    this.close();
    if(!this.childExit)return;
    if(this.closeWait)return this.closeWait;
    const child=this.shutdownChild;
    const waitWithin=async(ms)=>{
      let timer;
      try{return await Promise.race([this.childExit.then(()=>true),new Promise(ok=>{
        timer=setTimeout(()=>ok(false),ms);timer.unref?.();
      })]);}
      finally{clearTimeout(timer);}
    };
    this.closeWait=(async()=>{
      if(await waitWithin(graceMs))return;
      if(child && child.exitCode===null && child.signalCode===null)child.kill('SIGKILL');
      // Even after escalating, do not accept release until the OS reports 'close'.
      if(!await waitWithin(killMs))
        throw new NativeProtocolError('DAEMON_SHUTDOWN_TIMEOUT','Daemon closure not confirmed; native state remains unknown');
      throw new NativeProtocolError('DAEMON_SHUTDOWN_ESCALATED','Daemon required SIGKILL; native state must be reviewed');
    })();
    return this.closeWait;
  }
}
