import { NativeVgrObserver } from './index.mjs';

/**
 * Read-only, authoritative COMMIT cross-check through voderd's state machine.
 * Never accepts model output or caller-supplied strings as final evidence.
 * The producer must release the daemon's exclusive state lock before verification.
 */
export class NativeCommitVerificationError extends Error {
  constructor(code, detail) { super(detail); this.name='NativeCommitVerificationError'; this.code=code; }
}
const HEX64 = /^[0-9a-f]{64}$/;
const IDENT = /^[a-zA-Z0-9._:-]{1,128}$/;
function demand(ok, code, explanation) {
  if (!ok) throw new NativeCommitVerificationError(code, explanation);
}

export class NativeVgrCommitVerifier {
  constructor({ binary, stateDir, timeoutMs=12000, observerFactory }={}) {
    this.binary = binary;
    this.stateDir = stateDir;
    this.timeoutMs = timeoutMs;
    this.observerFactory = observerFactory ?? (()=>new NativeVgrObserver({binary, stateDir, timeoutMs}));
  }
  async confirmCommit(receipt, input) {
    demand(receipt && typeof receipt==='object' && input && typeof input==='object', 'BAD_INPUT','Missing receipt or request');
    demand(receipt.status==='COMMITTED', 'NOT_COMMITTED','Non-committed receipt');
    demand(receipt.session_id===input.session_id && receipt.action_id===input.action_id,'IDENTITY_MISMATCH','Receipt identity mismatch');
    demand(IDENT.test(receipt.run_id??''),'INVALID_RUN_ID','Receipt run identity missing or invalid');
    demand(HEX64.test(receipt.commit_digest??''),'INVALID_DIGEST','Final-commit identity missing or invalid');
    let observer;
    try {
      observer=await this.observerFactory();
      demand(observer && typeof observer.start==='function' && typeof observer.resume==='function' &&
        typeof observer.state==='function' && typeof observer.replay==='function' && typeof observer.close==='function',
        'UNSAFE_OBSERVER','Read-only Rust daemon observer required');
      await observer.start();
      const resumed=await observer.resume(receipt.session_id);
      demand(resumed.session_id===receipt.session_id && resumed.status==='READY','SESSION_MISMATCH','Rust daemon session mismatch');
      const state=await observer.state(receipt.run_id);
      demand(state.session_id===receipt.session_id,'SESSION_MISMATCH','Rust state mismatch');
      demand(Array.isArray(state.runs) && state.runs.length===1,'RUN_MISMATCH','Run not uniquely resolved');
      const run=state.runs[0];
      demand(run.run_id===receipt.run_id && run.status==='VGR_COMMITTED' && run.delivery_allowed===true &&
        run.final_commit_digest===receipt.commit_digest,
        'NO_AUTHORITATIVE_COMMIT','Native journal does not confirm the claimed final commit');
      // The public state view deliberately omits action_id. Bind it via the
      // independently replayed, daemon-validated vgr.final_committed event.
      let after=0, match=0, complete=false;
      for (let page=0;page<16;page++) {
        const answer=await observer.replay(receipt.run_id,after,256);
        demand(Array.isArray(answer.events),'MALFORMED_REPLAY','Malformed Rust journal replay');
        for (const row of answer.events) {
          demand(Number.isSafeInteger(row.sequence) && row.sequence>after && row.event &&
            typeof row.event==='object','MALFORMED_REPLAY','Invalid journal sequence/event');
          after=row.sequence;
          const e=row.event;
          if(e.type==='vgr.final_committed') {
            match++;
            demand(e.session===receipt.session_id && e.run_id===receipt.run_id &&
              e.action_id===receipt.action_id && e.commit_digest===receipt.commit_digest,
              'COMMIT_EVENT_MISMATCH','Committed event identity mismatch');
          }
        }
        if(answer.events.length<256){complete=true;break;}
      }
      demand(complete,'REPLAY_LIMIT','Journal replay exceeds bounded verification window');
      demand(match===1,'COMMIT_EVENT_MISSING','No unique authoritative commit event');
      return Object.freeze({ verified:true, authority:'VODER_NATIVE_VGR_STATE',
        session_id:receipt.session_id, run_id:receipt.run_id,
        action_id:receipt.action_id, commit_digest:receipt.commit_digest });
    } catch (e) {
      if(e instanceof NativeCommitVerificationError)throw e;
      throw new NativeCommitVerificationError('NATIVE_OBSERVATION_FAILED',e?.message??'Native observation failure');
    } finally {
      // Never report success until the independent Rust verifier has released its lock.
      if(observer){
        if(typeof observer.closeAndWait==='function')await observer.closeAndWait();
        else observer.close(); // Legacy injected observers implement only close().
      }
    }
  }
}
