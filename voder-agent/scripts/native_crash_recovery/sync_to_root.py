#!/usr/bin/env python3
"""Fail-closed install of validated T08 *test scripts*, only after real Rust crash PASS.

Never modifies Pi, VGR, Rust, SDK, CLI, or original test packages.
"""
import argparse, hashlib, importlib.util, json, os, re, shutil, stat, tempfile
from pathlib import Path

FILES = ('README_ZH.md','common.py','run_t08_only.sh','sync_to_root.py',
         'test_cargo_proxy.py','test_native_crash.py','test_node_realpath.py','SHA256SUMS')
PHASES = ('POST_PATCH','POST_META_AUDIT')

def h(b): return hashlib.sha256(b).hexdigest()

def plain(path, kind='file'):
    path=Path(path)
    st=path.lstat()
    if stat.S_ISLNK(st.st_mode):raise RuntimeError('SYMLINK_REJECTED:'+str(path))
    if kind=='file' and not stat.S_ISREG(st.st_mode):raise RuntimeError('NOT_FILE:'+str(path))
    if kind=='dir' and not stat.S_ISDIR(st.st_mode):raise RuntimeError('NOT_DIR:'+str(path))
    return path

def validated_source(source):
    source=plain(source,'dir')
    manifest=plain(source/'SHA256SUMS').read_text()
    entries={}
    for line in manifest.splitlines():
        m=re.fullmatch(r'([0-9a-f]{64})  ([A-Za-z0-9_.-]+)',line)
        if not m or m[2] in entries or m[2]=='SHA256SUMS':
            raise RuntimeError('BAD_SOURCE_MANIFEST')
        entries[m[2]]=m[1]
    if set(entries)!=set(FILES)-{'SHA256SUMS'}:
        raise RuntimeError('SOURCE_MANIFEST_FILESET_DRIFT')
    for name, expected in entries.items():
        if h(plain(source/name).read_bytes())!=expected:
            raise RuntimeError('SOURCE_HASH_DRIFT:'+name)
    return source

def validated_evidence(evidence):
    evidence=plain(evidence,'dir')
    if evidence.name.startswith('.') is False or not evidence.name.startswith('.voder-t08-node-realpath-'):
        raise RuntimeError('UNEXPECTED_EVIDENCE_DIR')
    report=json.loads(plain(evidence/'CRASH.json').read_text())
    if report.get('schema')!='voder.native.crash-rollback.v1' or report.get('pass') is not True:
        raise RuntimeError('T08_NOT_CONFIRMED_PASS')
    if report.get('model_called') is not False or report.get('automatic_sdk_resume')!='NOT_SUPPORTED':
        raise RuntimeError('INVALID_AUTHORITY_CLAIM')
    cases=report.get('cases')
    if not isinstance(cases,list) or len(cases)!=2 or {x.get('phase') for x in cases}!=set(PHASES):
        raise RuntimeError('INCOMPLETE_CRASH_PHASES')
    for case in cases:
        phase=case['phase']
        if (case.get('status')!='PASS' or case.get('sigkill_verified') is not True or
            case.get('recovery')!='RUST_MEASURED_ROLLBACK' or
            case.get('worktree_restored') is not True or
            case.get('original_repository_clean') is not True or
            case.get('sdk_auto_resume')!='NOT_SUPPORTED'):
            raise RuntimeError('INCOMPLETE_NATIVE_EVIDENCE:'+phase)
        sha=case.get('journal_sha256','')
        if not re.fullmatch(r'[0-9a-f]{64}',sha):
            raise RuntimeError('BAD_JOURNAL_DIGEST:'+phase)
        parent=plain(evidence/('crash-'+phase.lower()),'dir')
        matches=[p for p in parent.iterdir() if p.name.startswith('voder-vgr6-real-modal-')]
        if len(matches)!=1:raise RuntimeError('STATE_DIRECTORY_CARDINALITY:'+phase)
        state=plain(matches[0],'dir')
        journal=plain(state/'daemon-state/events.jsonl').read_bytes()
        if h(journal)!=sha:raise RuntimeError('JOURNAL_DIGEST_MISMATCH:'+phase)
        parsed=[json.loads(row)['body'] for row in journal.decode().splitlines() if row]
        recovered=[evt for evt in parsed if evt.get('type')=='vgr.write_reconciled']
        if len(recovered)!=1 or any(evt.get('type')=='vgr.final_committed' for evt in parsed):
            raise RuntimeError('RECOVERY_EVENTS_INCONSISTENT:'+phase)
        evt=recovered[0]
        if (evt.get('session')!='vgr6-accept-session' or evt.get('run_id')!='vgr6-accept-run' or
            evt.get('action_id')!='vgr6-accept-action' or evt.get('recovery_outcome')!='ROLLED_BACK'):
            raise RuntimeError('RECOVERY_IDENTITY_MISMATCH:'+phase)
    return report

def install_sources(source,root):
    source=validated_source(source)
    root=plain(root,'dir')
    parent=plain(root/'voder-agent/scripts','dir')
    dest=parent/'native_crash_recovery'
    if dest.exists() or dest.is_symlink():
        plain(dest,'dir')
        actual=set(p.name for p in dest.iterdir())
        if actual!=set(FILES):raise RuntimeError('ROOT_TEST_SCRIPT_DRIFT_FILESET')
        for file in FILES:
            if h(plain(dest/file).read_bytes())!=h(plain(source/file).read_bytes()):
                raise RuntimeError('ROOT_TEST_SCRIPT_DRIFT:'+file)
        print('ROOT_T08_SCRIPT_SYNC=PASS ALREADY_IDENTICAL',flush=True)
        return dest
    tmp=Path(tempfile.mkdtemp(prefix='.voder-t08-stage-',dir=parent))
    try:
        for file in FILES:
            destfile=tmp/file
            with destfile.open('xb') as o:o.write(plain(source/file).read_bytes());o.flush();os.fsync(o.fileno())
            destfile.chmod(0o755 if file=='run_t08_only.sh' else 0o644)
        if dest.exists() or dest.is_symlink():raise RuntimeError('ROOT_INSTALL_RACE')
        os.rename(tmp,dest)
    except BaseException:
        if tmp.exists():shutil.rmtree(tmp)
        raise
    for file in FILES:
        if h(plain(dest/file).read_bytes())!=h(plain(source/file).read_bytes()):
            raise RuntimeError('ROOT_INSTALL_VERIFY_FAILED:'+file)
    if not os.access(dest/'run_t08_only.sh',os.X_OK):raise RuntimeError('ROOT_RUNNER_NOT_EXECUTABLE')
    print('ROOT_T08_SCRIPT_SYNC=PASS INSTALLED',flush=True)
    return dest

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--evidence',required=True)
    parser.add_argument('--source',required=True)
    args=parser.parse_args()
    root=Path('/root/autodl-tmp')
    evidence=Path(args.evidence)
    if evidence.parent!=root/'repair':raise RuntimeError('EVIDENCE_NOT_UNDER_REPAIR')
    validated_source(args.source)
    validated_evidence(evidence)
    spec=importlib.util.spec_from_file_location('voder_common',Path(args.source)/'common.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if module.ROOT!=root or module.check_sources() is not True:raise RuntimeError('ROOT_SOURCE_PIN_CHECK_FAILED')
    dest=install_sources(args.source,root)
    # Test code only, no Rust/SDK/CLI product modules changed.
    module.check_sources()
    print('ROOT_T08_SCRIPT_DIR='+str(dest))
    print('VODER_PRODUCT_NATIVE_SOURCE_UNCHANGED=TRUE')
    print('T08_SDK_AUTOMATIC_SAFE_CONTINUATION=NOT_SUPPORTED')

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('ROOT_T08_SCRIPT_SYNC=FAILED '+str(e),flush=True)
        raise SystemExit(4)
