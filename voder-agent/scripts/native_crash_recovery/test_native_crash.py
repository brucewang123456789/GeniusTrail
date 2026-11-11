#!/usr/bin/env python3
"""Real SIGKILL of dedicated, controlled Rust test binary at durable checkpoint.
Not a model task, and not an SDK automatic resume; proves Rust native safe rollback.
"""
from common import *
from datetime import datetime, timezone
import signal, sys, time

check_sources()
EVID=checked(os.environ['VODER_TEST_EVIDENCE'],'dir')
report={'schema':'voder.native.crash-rollback.v1','at':datetime.now(timezone.utc).isoformat(),
 'model_called':False,'automatic_sdk_resume':'NOT_SUPPORTED','cases':[],'pass':False}
phases=os.environ.get('VODER_CRASH_PHASES','POST_PATCH,POST_META_AUDIT').split(',')
allowed={'POST_PATCH','POST_OBSERVED','POST_SETTLED','POST_ADMITTED','POST_META_AUDIT'}
if not phases or len(phases)!=len(set(phases)) or any(p not in allowed for p in phases):raise RuntimeError('INVALID_CRASH_PHASES')
if os.environ.get('VODER_ALLOW_MODAL_CHARGES')!='YES':raise RuntimeError('REAL_MODAL_CONSENT_REQUIRED')
if os.environ.get('VODER_ENABLE_NATIVE_SIGKILL')!='YES':raise RuntimeError('EXPLICIT_SIGKILL_CONSENT_REQUIRED')
node=trusted_regular_node(os.environ.get('VODER_NODE_BINARY',ROOT/'.voder-tools/node/bin/node'))
modal=checked(os.environ.get('VODER_MODAL_CLI',ROOT/'packages/voder-sandbox-modal/src/cli.ts'))
if not modal.is_absolute(): raise RuntimeError('MODAL_CLI_NOT_ABSOLUTE')
print('T08_RUST_NODE_REALPATH=PASS plain_executable='+str(node),flush=True)
print('T08_MODAL_CLI_REGULAR=PASS '+str(modal),flush=True)
example=checked(ROOT/'crates/voderd/examples/vgr_modal_write_closure.rs')
gateway=rust_binary(ROOT/'.voder-build/release/voder-gateway')
if b'fault_checkpoint(base, "POST_PATCH")?' not in example.read_bytes() or b'fault_checkpoint(base, "POST_META_AUDIT")?' not in example.read_bytes():
    raise RuntimeError('PINNED_CRASH_CHECKPOINTS_MISSING')
if b'"--recover"' not in checked(ROOT/'voder-agent/crates/voder-gateway/src/main.rs').read_bytes():
    raise RuntimeError('OPERATOR_RECOVERY_INTERFACE_MISSING')
# Build the CURRENT Rust test example once, without modifying sources or executing a model.
cargo=ROOT/'.voder-tools/rust/cargo/bin/cargo'
cargo=checked_toolchain_executable(cargo)
target_dir=ROOT/'.voder-build'
print('NATIVE_CRASH_EXAMPLE_BUILD=START cargo --locked --offline --release',flush=True)
cmd=[str(cargo),'build','--locked','--offline','--release','-p','voderd','--example','vgr_modal_write_closure','--target-dir',str(target_dir)]
compiled=run(cmd,timeout=900,env={**os.environ,'CARGO_TARGET_DIR':str(target_dir)})
(EVID/'rust-build.log').write_text((compiled.stdout+'\n'+compiled.stderr)[-15000:])
if compiled.returncode!=0:
    print('T08_NATIVE_RUST_CRASH_ROLLBACK=BLOCKED_CARGO_BUILD_FAILED (see rust-build.log)',flush=True)
    report['blocked']='CARGO_BUILD_FAILED';atomic_report(EVID/'CRASH.json',report);raise SystemExit(4)
binary=rust_binary(target_dir/'release/examples/vgr_modal_write_closure')
print('NATIVE_CRASH_EXAMPLE_BUILD=PASS',flush=True)

for phase in phases:
    case={'phase':phase,'status':'FAIL'};report['cases'].append(case)
    parent=EVID/('crash-'+phase.lower())
    parent.mkdir(mode=0o700)
    with (parent/'stdout.log').open('w') as out, (parent/'stderr.log').open('w') as err:
        env={**os.environ,'VODER_NODE_BINARY':str(node),'VODER_MODAL_CLI':str(modal),
             'VODER_VGR6_EVIDENCE_DIR':str(parent),'VODER_VGR_FAULT_PAUSE_AFTER':phase}
        # Direct binary; only this known test PID is ever killed. No shell/process groups.
        child=subprocess.Popen([str(binary)],stdout=out,stderr=err,env=env,stdin=subprocess.DEVNULL)
        state=None
        deadline=time.monotonic()+int(os.environ.get('VODER_FAULT_WAIT_SECONDS','360'))
        try:
            while time.monotonic()<deadline:
                candidates=[p for p in parent.iterdir() if p.is_dir() and p.name.startswith('voder-vgr6-real-modal-')]
                if len(candidates)>1:raise RuntimeError('AMBIGUOUS_NATIVE_PROCESS_ROOT')
                if candidates:
                    state=candidates[0]
                    marker=state/('checkpoint-'+phase+'.ready')
                    if marker.is_file():break
                if child.poll() is not None:raise RuntimeError('RUST_PROCESS_EXITED_BEFORE_FAULT: '+(parent/'stderr.log').read_text()[-850:])
                time.sleep(.2)
            else:raise RuntimeError('RUST_CHECKPOINT_TIMEOUT')
            if marker.read_text()!=phase:raise RuntimeError('UNTRUSTED_FAULT_MARKER')
            # Prove we are killing the exact child we started at the durable checkpoint.
            if child.poll() is not None:raise RuntimeError('PROCESS_ALREADY_EXITED')
            child.send_signal(signal.SIGKILL)
            rc=child.wait(timeout=10)
            if rc!=-signal.SIGKILL:raise RuntimeError('NOT_SIGKILL_TERMINATED: '+str(rc))
            case['sigkill_verified']=True
            # The example holds a detached worktree, never production sources.
            state=owned(checked(state,'dir'),parent)
            repo=owned(checked(state/'original-git-repository','dir'),state)
            worktree=owned(checked(state/'isolated-worktrees/verified-write','dir'),state)
            daemon=owned(checked(state/'daemon-state','dir'),state)
            baseline=checked(repo/'src/input.txt')
            mutated=checked(worktree/'src/input.txt')
            if baseline.read_bytes()!=b'VODER_VGR_REAL_MODAL_WRITE_OLD\n':raise RuntimeError('BASELINE_CHANGED')
            if mutated.read_bytes()!=b'VODER_VGR_REAL_MODAL_WRITE_VERIFIED\n':raise RuntimeError('CRASH_NOT_AFTER_PATCH')
            before_events=events(daemon/'events.jsonl')
            if any(e.get('type')=='vgr.final_committed' for e in before_events):raise RuntimeError('COMMITTED_BEFORE_INTENDED_CRASH')
            case['interrupted_stage']=next((e.get('type') for e in reversed(before_events) if e.get('type','').startswith('vgr.')),None)
            # Operator-only Rust rollback; NEVER auto-submit or replay a tool request.
            recovery=run([gateway,'--recover',daemon,'vgr6-accept-session','vgr6-accept-run','vgr6-accept-action'],timeout=40)
            if recovery.returncode!=0 or 'VODER_NATIVE_RECOVERY=RolledBack' not in recovery.stderr:
                raise RuntimeError('RUST_RECOVERY_FAILED: '+recovery.stderr[-650:])
            if mutated.read_bytes()!=b'VODER_VGR_REAL_MODAL_WRITE_OLD\n':raise RuntimeError('ROLLBACK_BYTES_MISMATCH')
            if baseline.read_bytes()!=b'VODER_VGR_REAL_MODAL_WRITE_OLD\n':raise RuntimeError('ORIGINAL_REPO_MODIFIED')
            git=run(['/usr/bin/git','-C',repo,'status','--porcelain'],timeout=10)
            if git.returncode!=0 or git.stdout.strip():raise RuntimeError('ORIGINAL_REPO_DIRTY')
            after_events=events(daemon/'events.jsonl')
            reconciled=[e for e in after_events if e.get('type')=='vgr.write_reconciled']
            if len(reconciled)!=1 or any(e.get('type')=='vgr.final_committed' for e in after_events):
                raise RuntimeError('RECOVERY_JOURNAL_AUTHORITY_MISMATCH')
            evt=reconciled[0]
            if evt.get('session')!='vgr6-accept-session' or evt.get('run_id')!='vgr6-accept-run' or evt.get('action_id')!='vgr6-accept-action' or evt.get('recovery_outcome')!='ROLLED_BACK':
                raise RuntimeError('RECOVERY_IDENTITY_MISMATCH')
            response=evt.get('response')
            if not isinstance(response,dict) or response.get('status')!='VGR_RECOVERED_ROLLBACK' or response.get('delivery_allowed') is not False or response.get('execution_authorized') is not False:
                raise RuntimeError('RECOVERY_WOULD_GRANT_DELIVERY')
            once_hash=digest(checked(daemon/'events.jsonl').read_bytes())
            second=run([gateway,'--recover',daemon,'vgr6-accept-session','vgr6-accept-run','vgr6-accept-action'],timeout=25)
            if second.returncode==0 or 'run not eligible for interrupted-write rollback' not in second.stderr:
                raise RuntimeError('DUPLICATE_RECOVERY_NOT_REJECTED: '+second.stderr[-320:])
            if digest(checked(daemon/'events.jsonl').read_bytes())!=once_hash:raise RuntimeError('DUPLICATE_RECOVERY_MUTATED_JOURNAL')
            case.update({'status':'PASS','journal_sha256':once_hash,'recovery':'RUST_MEASURED_ROLLBACK',
              'worktree_restored':True,'original_repository_clean':True,'sdk_auto_resume':'NOT_SUPPORTED'})
            print(f'T08_NATIVE_CRASH_{phase}=PASS SIGKILL + Rust rollback + durable non-delivery + duplicate recovery rejected',flush=True)
        except Exception as e:
            case['error']=str(e)[:850];print(f'T08_NATIVE_CRASH_{phase}=FAIL {e}',flush=True)
        finally:
            if child.poll() is None:
                child.kill();child.wait(timeout=10)

report['pass']=len(report['cases'])==len(phases) and all(c['status']=='PASS' for c in report['cases'])
atomic_report(EVID/'CRASH.json',report)
print('T08_NATIVE_RUST_CRASH_ROLLBACK='+('PASS' if report['pass'] else 'FAIL'))
print('T08_SDK_AUTOMATIC_SAFE_CONTINUATION=NOT_SUPPORTED (this suite tests operator Rust rollback, not SDK auto-resume)')
raise SystemExit(0 if report['pass'] else 4)
