import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,access,chmod} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {spawn} from 'node:child_process';
import {NativeVgrObserver} from '../../voder-native-bridge/src/index.mjs';
import {NativeSdkContinuation} from '../src/native-continuation.mjs';

const ACTION='action-lock-proof';
const SESSION='session-lock-proof';
const RUN=`run-${ACTION}`;
const DIGEST='a'.repeat(64);
// This is an isolated Python stand-in for the Rust wire protocol. It enforces
// the REAL Linux flock operation on a temporary lockfile; it cannot commit code.
const fakeDaemon=`#!/usr/bin/env python3
import os,sys,json,fcntl,signal
state=sys.argv[sys.argv.index('--state-dir')+1]
fd=os.open(os.path.join(state,'daemon.lock'),os.O_CREAT|os.O_RDWR,0o600)
try: fcntl.flock(fd, fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:
 with open(os.path.join(state,'lock_denied'),'a') as f: f.write('EAGAIN\\n')
 sys.stderr.write('LOCK_DENIED_EAGAIN\\n');sys.exit(1)
if os.environ.get('VODER_TEST_IGNORE_TERM')=='1':signal.signal(signal.SIGTERM,signal.SIG_IGN)
if os.environ.get('VODER_TEST_EXIT_IMMEDIATELY')=='1':sys.exit(1)
for line in sys.stdin:
 try:
  req=json.loads(line)
  kind=req['kind'];session=req.get('session_id','${SESSION}')
  if kind=='initialize': payload={'protocol_version':1}
  elif kind=='session.resume':payload={'session_id':session,'status':'READY'}
  elif kind=='state.get':payload={'session_id':session,'runs':[{'run_id':'${RUN}','status':'VGR_COMMITTED','delivery_allowed':True,'final_commit_digest':'${DIGEST}'}]}
  elif kind=='run.replay':payload={'events':[{'sequence':1,'event':{'type':'vgr.final_committed','session':'${SESSION}','run_id':'${RUN}','action_id':'${ACTION}','commit_digest':'${DIGEST}'}}] if req['payload'].get('after_sequence',0)==0 else []}
  else:payload={'code':'DENIED','message':'Not an observer operation'}
  print(json.dumps({'protocol_version':1,'frame_type':'response','request_id':req['request_id'],'kind':kind,'payload':payload}),flush=True)
 except Exception as e:
  sys.stderr.write(repr(e)+'\\n');sys.exit(2)
`;
async function fixture(){
 const root=await mkdtemp(join(tmpdir(),'voder-native-lock-test-'));
 const store=join(root,'store');
 const stateDir=join(store,'attempts',ACTION,'daemon-state');
 await mkdir(stateDir,{recursive:true});
 const daemon=join(root,'fake-voderd.py');
 await writeFile(daemon,fakeDaemon,{mode:0o700});
 await chmod(daemon,0o700);
 const action={session_id:SESSION,run_id:RUN,action_id:ACTION};
 return {root,store,stateDir,daemon,action};
}
const observer=f=>new NativeVgrObserver({binary:f.daemon,stateDir:f.stateDir,timeoutMs:3000});

// Failing before the fix: a second Rust daemon cannot obtain LOCK_EX while the
// first observer remains alive. This test proves the precondition with OS flock.
test('OS lock: simultaneous observer is rejected; shutdown barrier restores availability',async()=>{
 const f=await fixture(),first=observer(f);
 await first.start();await first.resume(SESSION);
 const contender=observer(f);
 await assert.rejects(contender.start(), e=>e?.code==='DAEMON_EXIT'||e?.code==='WRITE_ERROR');
 assert.match(await readFile(join(f.stateDir,'lock_denied'),'utf8'),/EAGAIN/);
 await first.closeAndWait();
 const later=observer(f);
 try{await later.start();await later.resume(SESSION);assert.equal((await later.state(RUN)).runs[0].status,'VGR_COMMITTED');}
 finally{await later.closeAndWait();}
});

test('SDK committed observation waits for first daemon close before independent verifier',async()=>{
 const f=await fixture();
 const recovery=new NativeSdkContinuation({gatewayBinary:f.daemon,daemonBinary:f.daemon,attemptStore:f.store,timeoutMs:3000});
 const r=await recovery._observe(f.action);
 assert.equal(r.status,'COMMITTED');assert.equal(r.commit_digest,DIGEST);
 await assert.rejects(access(join(f.stateDir,'lock_denied')), {code:'ENOENT'});
});

test('unverified historical commit never calls Rust rollback',async()=>{
 const f=await fixture();
 const recovery=new NativeSdkContinuation({gatewayBinary:join(f.root,'must-not-run'),daemonBinary:f.daemon,attemptStore:f.store,timeoutMs:3000});
 recovery.verifier.confirmCommit=async()=>{throw Object.assign(Error('negative cross-check'),{code:'MISMATCH'});};
 const r=await recovery.reconcile(f.action);
 assert.equal(r.status,'UNKNOWN');assert.equal(r.reason,'COMMITTED_BUT_UNVERIFIED');
});

test('pipe closed before initialize fails without uncaught EPIPE',async()=>{
 const f=await fixture();
 const old=process.env.VODER_TEST_EXIT_IMMEDIATELY;
 process.env.VODER_TEST_EXIT_IMMEDIATELY='1';
 try{
  const o=observer(f);
  await assert.rejects(o.start(),e=>e?.code==='DAEMON_EXIT'||e?.code==='WRITE_ERROR');
  await o.closeAndWait();
 }finally{
  if(old===undefined)delete process.env.VODER_TEST_EXIT_IMMEDIATELY;
  else process.env.VODER_TEST_EXIT_IMMEDIATELY=old;
 }
});

test('uncooperative daemon is SIGKILLed but shutdown is fail-closed',async()=>{
 const f=await fixture();
 const old=process.env.VODER_TEST_IGNORE_TERM;
 process.env.VODER_TEST_IGNORE_TERM='1';
 try{
  const o=observer(f);await o.start();await o.resume(SESSION);
  await assert.rejects(o.closeAndWait({graceMs:150,killMs:1500}),e=>e?.code==='DAEMON_SHUTDOWN_ESCALATED');
  // Post-SIGKILL, the state lock is freed; verification remains blocked, not falsely successful.
  delete process.env.VODER_TEST_IGNORE_TERM;
  const next=observer(f);try{await next.start();}finally{await next.closeAndWait();}
 }finally{
  if(old===undefined)delete process.env.VODER_TEST_IGNORE_TERM;
  else process.env.VODER_TEST_IGNORE_TERM=old;
 }
});
