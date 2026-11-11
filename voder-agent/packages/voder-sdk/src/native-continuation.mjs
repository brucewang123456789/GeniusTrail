/** Operator-owned Rust reconciliation. Never a model tool, never commit authority. */
import {spawn} from 'node:child_process';
import {lstat} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {NativeVgrObserver} from '../../voder-native-bridge/src/index.mjs';
import {GatewayCommitVerifier} from '../../voder-native-bridge/src/trusted-gateway.mjs';
import {VoderError} from '../../voder-agent-core/src/contracts.mjs';
const HEX=/^[a-f0-9]{64}$/;
const ID=/^[a-zA-Z0-9._-]{1,64}$/;
const error=(code,msg)=>new VoderError(code,msg);

async function runRecovery(binary,args,timeoutMs){
  return new Promise((done,fail)=>{
    const child=spawn(binary,args,{shell:false,stdio:['ignore','ignore','ignore']});
    let settled=false;
    const finish=(err)=>{if(settled)return;settled=true;clearTimeout(timer);err?fail(err):done();};
    const timer=setTimeout(()=>{child.kill('SIGKILL');finish(error('RUST_RECOVERY_TIMEOUT','Operator Rust recovery timed out'));},timeoutMs);
    child.on('error',e=>finish(error('RUST_RECOVERY_SPAWN',e.message)));
    child.on('close',(code,signal)=>finish(code===0&&!signal?null:error('RUST_RECOVERY_REJECTED','Rust recovery refused; cannot auto-continue')));
  });
}

export class NativeSdkContinuation {
  constructor({gatewayBinary,daemonBinary,attemptStore,timeoutMs=12000,rollbackTimeoutMs=60000}={}){
    if([gatewayBinary,daemonBinary,attemptStore].some(x=>typeof x!=='string'||!x.startsWith('/')))
      throw error('RECOVERY_CONFIG','Operator-pinned absolute gateway, daemon and attempt store required');
    if(!Number.isInteger(timeoutMs)||timeoutMs<1000||timeoutMs>120000)throw error('RECOVERY_TIMEOUT','Observer timeout must be in native bounds');
    if(!Number.isInteger(rollbackTimeoutMs)||rollbackTimeoutMs<1000||rollbackTimeoutMs>120000)throw error('RECOVERY_TIMEOUT','Rollback timeout invalid');
    this.rollbackTimeoutMs=rollbackTimeoutMs;this.gatewayBinary=resolve(gatewayBinary);this.daemonBinary=resolve(daemonBinary);this.attemptStore=resolve(attemptStore);this.timeoutMs=timeoutMs;
    this.verifier=new GatewayCommitVerifier({daemonBinary:this.daemonBinary,attemptStore:this.attemptStore,timeoutMs});
  }
  async _observe(action){
    const stateDir=join(this.attemptStore,'attempts',action.action_id,'daemon-state');
    const st=await lstat(stateDir);if(st.isSymbolicLink()||!st.isDirectory())throw error('UNSAFE_RUST_STATE','Unsafe native state dir');
    const observer=new NativeVgrObserver({binary:this.daemonBinary,stateDir,timeoutMs:this.timeoutMs});
    let nativeCommit=null;
    let sawCommitted=false;
    let result;
    try{
      await observer.start();await observer.resume(action.session_id);
      const state=await observer.state(action.run_id);
      if(state.session_id!==action.session_id||state.runs.length!==1||state.runs[0]?.run_id!==action.run_id)
        throw error('RECOVERY_IDENTITY','Rust run/session mismatch');
      const run=state.runs[0];
      const events=[];let after=0,complete=false;
      for(let page=0;page<16;page++){
        const reply=await observer.replay(action.run_id,after,256);
        if(!Array.isArray(reply.events))throw error('RECOVERY_REPLAY','Invalid Rust event replay');
        for(const row of reply.events){if(!Number.isSafeInteger(row.sequence)||row.sequence<=after||!row.event)throw error('RECOVERY_REPLAY','Invalid event sequence');after=row.sequence;events.push(row.event);}
        if(reply.events.length<256){complete=true;break;}
      }
      if(!complete)throw error('RECOVERY_REPLAY_LIMIT','Incomplete Rust replay');
      if(run.status==='VGR_COMMITTED'){
        sawCommitted=true;
        const commits=events.filter(e=>e.type==='vgr.final_committed');
        if(commits.length!==1||commits[0].session!==action.session_id||commits[0].run_id!==action.run_id||commits[0].action_id!==action.action_id||!HEX.test(commits[0].commit_digest??''))
          throw error('COMMIT_IDENTITY','Native commit is not uniquely bound');
        const receipt={status:'COMMITTED',session_id:action.session_id,action_id:action.action_id,run_id:action.run_id,commit_digest:commits[0].commit_digest};
        // Rust uses an exclusive state lock. Preserve this receipt but DO NOT open
        // the independent verifier until this observer's OS process has exited.
        nativeCommit=receipt;
        result={status:'COMMITTED',action_id:action.action_id,authority:'VODER_NATIVE_VGR_STATE',commit_digest:receipt.commit_digest};
      } else if(run.status==='VGR_RECOVERED_ROLLBACK'){
        const eventsReconciled=events.filter(e=>e.type==='vgr.write_reconciled');
        if(eventsReconciled.length!==1||eventsReconciled[0].session!==action.session_id||eventsReconciled[0].run_id!==action.run_id||eventsReconciled[0].action_id!==action.action_id||run.delivery_allowed!==false||events.some(e=>e.type==='vgr.final_committed'))
          throw error('ROLLBACK_IDENTITY','Rollback not uniquely evidenced or was committed');
        result={status:'ROLLED_BACK',action_id:action.action_id,authority:'VODER_NATIVE_VGR_RECOVERY',delivery_allowed:false};
      } else {
        result={status:'UNKNOWN',action_id:action.action_id,reason:'Native run state cannot authorize continuation'};
      }
    } catch(e) {
      // Once Rust reports committed, NO recovery is permitted even on corrupt
      // replay/identity, broken pipe, or cleanup failure. Only manual review.
      if(sawCommitted)throw error('COMMITTED_BUT_UNVERIFIED',`Native committed state unverified: ${e?.code??'INVALID_EVIDENCE'}`);
      throw e;
    } finally {
      // Independent Rust journal verification MUST follow a confirmed 'close'.
      // Any failed/uncertain shutdown prevents a new observer and fails closed.
      try{await observer.closeAndWait();}
      catch(e){
        if(sawCommitted)throw error('COMMITTED_BUT_UNVERIFIED',`Native daemon shutdown unconfirmed: ${e?.code??'UNKNOWN'}`);
        throw e;
      }
    }
    if(nativeCommit){
      try {await this.verifier.confirmCommit(nativeCommit,action);}
      catch(e){throw error('COMMITTED_BUT_UNVERIFIED',`Committed native run could not be independently verified: ${e?.code??'NATIVE_ERROR'}`);}
    }
    return result;
  }
  async reconcile(action){
    if(!action||!ID.test(action.action_id??'')||!ID.test(action.session_id??'')||action.run_id!==`run-${action.action_id}`)
      throw error('INVALID_RECOVERY_IDENTITY','Untrusted action metadata');
    // Existing native commits are inspected first; never ask Rust to roll back a committed action.
    try{
      const first=await this._observe(action);
      if(first.status!=='UNKNOWN')return first;
    }catch(e){
      // A witnessed commit with failed verification must never trigger rollback.
      if(e?.code==='COMMITTED_BUT_UNVERIFIED')return {status:'UNKNOWN',action_id:action.action_id,reason:'COMMITTED_BUT_UNVERIFIED'};
      // Inflight journal might be unopenable until operator recovery; recover may still refuse.
    }
    const stateDir=join(this.attemptStore,'attempts',action.action_id,'daemon-state');
    const st=await lstat(stateDir).catch(()=>null);
    if(!st||!st.isDirectory()||st.isSymbolicLink())return {status:'UNKNOWN',action_id:action.action_id,reason:'No independent native state evidence'};
    try{await runRecovery(this.gatewayBinary,['--recover',stateDir,action.session_id,action.run_id,action.action_id],this.rollbackTimeoutMs);}catch(e){return {status:'UNKNOWN',action_id:action.action_id,reason:e?.code??'RUST_RECOVERY_REJECTED'};}
    try{return await this._observe(action);}catch(e){return {status:'UNKNOWN',action_id:action.action_id,reason:e?.code??'OBSERVATION_FAILED'};}
  }
}
