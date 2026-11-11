'use strict';
/** JSON-lines transport to SDK-CLI. UI is not an alternate Agent or VGR authority. */
const {spawn} = require('node:child_process');
const fs=require('node:fs');
const path=require('node:path');
const {EventEmitter} = require('node:events');
const {StringDecoder} = require('node:string_decoder');
const MAX_FRAME=require('./input-capacity.cjs').LIMITS.wireBytes;
const {validateReferences}=require('./reference-contract.cjs');
class RpcError extends Error {constructor(code,message){super(message);this.code=code;}}
class RpcClient extends EventEmitter {
  constructor({command='node',args,cwd,env=process.env,spawnFn=spawn,diagnosticFile=null}={}) {
    super();
    if(typeof command!=='string'||!command.trim()||!Array.isArray(args)||!args.every(x=>typeof x==='string')||typeof cwd!=='string'||!cwd)
      throw new RpcError('INVALID_CLIENT_CONFIG','Executable, arguments and workspace required');
    this.child=spawnFn(command,args,{cwd,env,stdio:['pipe','pipe','pipe'],shell:false,windowsHide:true});
    this.diagnosticFile=diagnosticFile;this.stderrChunks=[];this.stderrBytes=0;
    this.child.stdin.on('error',()=>{});
    this.alive=true;this.seq=0;this.pending=new Map();this.input='';this.decoder=new StringDecoder('utf8');
    // Task waits use a rolling per-event inactivity watchdog, never a task-total deadline.
    this.stallCancelRpcMs=15000;this.stallReconcileMs=60000;
    this.child.stdout.on('data',chunk=>this._read(chunk));
    // The CLI uses stderr for diagnostics only: never include raw stderr in UI logs.
    this.child.stderr.on('data',chunk=>{if(this.stderrBytes>=8192)return;const bytes=Buffer.from(chunk).subarray(0,8192-this.stderrBytes);this.stderrChunks.push(bytes);this.stderrBytes+=bytes.length;});
    this.child.on('error',err=>{this._recordDiagnostic('SPAWN',err.code||'UNKNOWN');this._stop(new RpcError('RPC_SPAWN_FAILED',err.code||'Voder CLI process failed to start'));});
    this.child.on('exit',(code,signal)=>{this._recordDiagnostic('EXIT',`code=${code??'null'} signal=${signal??'none'}`);this._stop(new RpcError('RPC_EXIT',`Voder CLI closed (exit=${code??'null'}, signal=${signal??'none'})`));});
  }
  _recordDiagnostic(stage,detail){
    if(!this.diagnosticFile)return;
    try{
      // Never persist raw stderr. Only allowlisted diagnostic lines are retained.
      const source=Buffer.concat(this.stderrChunks).toString('utf8');
      const lines=source.split(/[\r\n]+/).filter(Boolean).slice(-24);
      const safe=lines.map(line=>{
        let s=line.replace(/\x1b\[[0-9;]*m/g,'').slice(0,360);
        for(const key of ['MODAL_TOKEN_ID','MODAL_TOKEN_SECRET','OPENAI_API_KEY','ANTHROPIC_API_KEY']){
          const secret=process.env[key];if(secret&&secret.length>5)s=s.split(secret).join('[REDACTED]');
        }
        s=s.replace(/(Bearer\s+)[^\s]+/gi,'$1[REDACTED]')
           .replace(/((?:token|secret|password|api[_-]?key|authorization)\s*[:=]\s*)[^\s,;]+/gi,'$1[REDACTED]')
           .replace(/\b(?:sk|pk|rk)-[A-Za-z0-9_-]{8,}\b/g,'[REDACTED]');
        // Only errors and stack frames, excluding arbitrary stdout/user content.
        return /(?:error|exception|failed|not found|cannot find|ENOENT|EACCES|MODULE_NOT_FOUND|at \S|stack|invalid)/i.test(s)?s:null;
      }).filter(Boolean).slice(-12);
      const report={timestamp:new Date().toISOString(),stage,detail:String(detail).slice(0,120),stderr:safe,stderr_truncated:this.stderrBytes>=8192};
      const dir=path.dirname(this.diagnosticFile);
      fs.mkdirSync(dir,{recursive:true,mode:0o700});
      const st=fs.lstatSync(dir);if(!st.isDirectory()||st.isSymbolicLink())return;
      fs.appendFileSync(this.diagnosticFile,JSON.stringify(report)+'\n',{encoding:'utf8',mode:0o600,flag:'a'});
      fs.chmodSync(this.diagnosticFile,0o600);
    }catch{/* Diagnostics must never affect VGR or RPC behavior. */}
  }
  _read(chunk){
    if(!this.alive)return;
    this.input+=this.decoder.write(chunk);
    if(Buffer.byteLength(this.input,'utf8')>MAX_FRAME*2){this.close('RPC_FRAME_TOO_LARGE');return;}
    while(true){const nl=this.input.indexOf('\n');if(nl<0)break;
      const line=this.input.slice(0,nl);this.input=this.input.slice(nl+1);
      if(Buffer.byteLength(line,'utf8')>MAX_FRAME){this.close('RPC_FRAME_TOO_LARGE');return;}
      let msg;try{msg=JSON.parse(line);}catch{this.close('RPC_BAD_JSON');return;}
      if(msg&&msg.jsonrpc==='2.0'&&msg.method==='session.event'){
        if(msg.params&&typeof msg.params.sessionId==='string'){
          this._taskProgress(msg.params);
          this.emit('event',msg.params);
        }
        continue;
      }
      if(msg?.jsonrpc!=='2.0'||!Number.isSafeInteger(msg?.id)) {this.close('RPC_BAD_RESPONSE');return;}
      const p=this.pending.get(msg.id);if(!p)continue;
      this.pending.delete(msg.id);clearTimeout(p.timer);
      if(msg.error)p.reject(new RpcError(String(msg.error.code||'RPC_ERROR'),String(msg.error.message||'RPC request failed')));
      else p.resolve(msg.result);
    }
  }
  _taskProgress({sessionId,event}){
    if(!event||typeof event.type!=='string')return;
    for(const [id,p] of this.pending){
      if(p.method!=='task.wait'||p.sessionId!==sessionId||p.stalled)continue;
      // Replay and duplicate events cannot masquerade as new forward progress.
      if(!Number.isSafeInteger(event.sequence)||event.sequence<=p.lastSequence)continue;
      p.lastSequence=event.sequence;p.stage=event.type;
      clearTimeout(p.timer);
      if(event.type==='task.workspace_approval_requested'){
        // A human approval dialog is not a fault. Resume on authorization dispatch.
        p.waitingForHuman=true;p.timer=null;
      }else{
        p.waitingForHuman=false;
        p.timer=setTimeout(()=>this._stageStalled(id),p.idleMs);
      }
    }
  }
  _stageStalled(id){
    const p=this.pending.get(id);
    if(!p||p.method!=='task.wait'||p.stalled||p.waitingForHuman)return;
    p.stalled=true;p.timer=null;
    // Silence beyond the configured stage interval justifies a cancellation
    // request, not an assertion that Rust/native execution has stopped.
    this.emit('task-stalled',{sessionId:p.sessionId,stage:p.stage,idleMs:p.idleMs});
    void this.request('session.cancel',{sessionId:p.sessionId},this.stallCancelRpcMs).then(ack=>{
      if(this.pending.get(id)!==p)return;
      if(ack?.sessionId!==p.sessionId||ack?.status!=='CANCELLATION_REQUESTED'){
        this._failStalledWait(id,'TASK_STALLED_CANCEL_UNCONFIRMED');return;
      }
      // The original task.wait still owns the final outcome. If control-plane
      // settlement never arrives, report uncertainty, not a fake cancellation.
      p.timer=setTimeout(()=>this._failStalledWait(id,'TASK_STALLED_RECONCILIATION_REQUIRED'),this.stallReconcileMs);
    }).catch(()=>this._failStalledWait(id,'TASK_STALLED_CANCEL_UNCONFIRMED'));
  }
  _failStalledWait(id,code){
    const p=this.pending.get(id);
    if(!p||p.method!=='task.wait')return;
    this.pending.delete(id);clearTimeout(p.timer);
    p.reject(new RpcError(code,`No new stage event for ${p.idleMs} ms; session ${p.sessionId} requires status/native outcome reconciliation. Task completion is not assumed.`));
  }
  request(method,params={},timeoutMs=20000){
    if(!this.alive)return Promise.reject(new RpcError('RPC_DISCONNECTED','Voder control process unavailable'));
    if(!/^[a-z][a-zA-Z]*(?:\.[a-zA-Z]+)*$/.test(method)||!params||typeof params!=='object'||Array.isArray(params))return Promise.reject(new RpcError('INVALID_RPC_REQUEST','Invalid method or params'));
    if(this.pending.size>=64)return Promise.reject(new RpcError('RPC_BACKPRESSURE','Too many pending calls'));
    const id=++this.seq,data=JSON.stringify({jsonrpc:'2.0',id,method,params})+'\n';
    let limit=65536;
    if(method==='task.submit'&&params.task?.referenceContext?.length){try{validateReferences(params.task.referenceContext);}catch(e){return Promise.reject(e);}if(this.referenceContextCapable!==true)return Promise.reject(new RpcError('REFERENCE_RUNTIME_UNSUPPORTED','Runtime does not support file context'));limit=Math.min(MAX_FRAME,Number.isSafeInteger(this.referenceFrameBytes)?this.referenceFrameBytes:MAX_FRAME);}
    if(Buffer.byteLength(data,'utf8')>limit)return Promise.reject(new RpcError('RPC_FRAME_TOO_LARGE','Request exceeds CLI frame policy'));
    if(!Number.isSafeInteger(timeoutMs)||timeoutMs<1)return Promise.reject(new RpcError('INVALID_RPC_TIMEOUT','A positive finite per-request interval is required'));
    const isWait=method==='task.wait';
    if(isWait&&(typeof params.sessionId!=='string'||!params.sessionId))return Promise.reject(new RpcError('INVALID_RPC_REQUEST','task.wait requires session identity'));
    return new Promise((resolve,reject)=>{
      const p={resolve,reject,timer:null,method,sessionId:isWait?params.sessionId:null,
        idleMs:isWait?timeoutMs:0,lastSequence:-1,stage:'task.wait',waitingForHuman:false,stalled:false};
      p.timer=setTimeout(()=>{
        if(isWait)this._stageStalled(id);
        else{this.pending.delete(id);reject(new RpcError('RPC_TIMEOUT',`${method} timed out; execution outcome unknown`));}
      },timeoutMs);
      this.pending.set(id,p);
      try{this.child.stdin.write(data);
        if(method==='task.authorizeWorkspace'&&typeof params.sessionId==='string'){
          // Do not time out a user while the approval dialog is open.
          // Once the decision is sent, the stage inactivity clock resumes.
          for(const [waitId,q] of this.pending)if(q.method==='task.wait'&&q.sessionId===params.sessionId&&q.waitingForHuman&&!q.stalled){
            q.waitingForHuman=false;q.stage='task.authorizeWorkspace';
            clearTimeout(q.timer);q.timer=setTimeout(()=>this._stageStalled(waitId),q.idleMs);
          }
        }
      }catch(e){clearTimeout(p.timer);this.pending.delete(id);reject(new RpcError('RPC_WRITE_FAILED',e.code||'RPC stdin unavailable'));}
    });
  }
  _stop(err){if(!this.alive)return;this.alive=false;
    for(const p of this.pending.values()){clearTimeout(p.timer);p.reject(err);}this.pending.clear();this.emit('disconnect',err);
  }
  close(reason='RPC_CLOSED'){
    if(!this.alive)return;
    this._stop(new RpcError(reason,'Voder RPC connection closed; in-flight effects must be reconciled by native VGR'));
    this.child.kill('SIGTERM');
  }
}
module.exports={RpcClient,RpcError};
