"""Shared fail-closed helpers for native acceptance; no product source writes."""
import hashlib, json, os, stat, subprocess
from pathlib import Path
ROOT = Path(os.environ.get('VODER_ROOT','/root/autodl-tmp')).resolve()
REPAIR = ROOT / 'repair'
# Source pins from the 2026-10-13 frozen Voder source snapshot; refuse divergent native code.
SOURCE_PINS = {
 'voder-agent/crates/voder-gateway/src/main.rs': 'c6418b4b022a13d766e31f01cbe51176b75e5898e5539bdd084d4525ce7e9a12',
 'crates/voderd/examples/vgr_modal_write_closure.rs': 'd4e8e355d3364f6b7c10781681a79a82c1e1780502a24f51708f59866dd030d6',
 'crates/voderd/src/native_write_recovery.rs': 'd0f35b248cf4d8901c3e5f96f0bbcdcd3c6bc11c22c6172378bca1bd37558ab1',
}
def check_sources():
    for rel, expected in SOURCE_PINS.items():
        actual=digest(checked(ROOT/rel).read_bytes())
        if actual != expected:raise RuntimeError(f'PINNED_SOURCE_DRIFT:{rel}:expected={expected}:got={actual}')
    return True

def digest(data): return hashlib.sha256(data).hexdigest()
def checked(p, kind='file'):
    p=Path(p)
    st=p.lstat()
    if stat.S_ISLNK(st.st_mode) or not (stat.S_ISREG(st.st_mode) if kind=='file' else stat.S_ISDIR(st.st_mode)):
        raise RuntimeError('NOT_PLAIN_'+kind.upper()+': '+str(p))
    return p

def checked_toolchain_executable(path):
    """Accept a symlinked toolchain executable, but only under the pinned tools tree.

    Other evidence/artifact paths still use checked() and reject symlinks.
    """
    p=Path(path)
    if not p.is_absolute(): raise RuntimeError('TOOLCHAIN_PATH_NOT_ABSOLUTE: '+str(p))
    tools=(ROOT/'.voder-tools').resolve(strict=True)
    try: resolved=p.resolve(strict=True)
    except (OSError,RuntimeError) as e: raise RuntimeError('TOOLCHAIN_RESOLVE_FAILED: '+str(p)) from e
    if resolved==tools or tools not in resolved.parents:
        raise RuntimeError('TOOLCHAIN_OUTSIDE_PINNED_ROOT: '+str(p))
    if not stat.S_ISREG(resolved.stat().st_mode) or not os.access(resolved,os.X_OK):
        raise RuntimeError('TOOLCHAIN_NOT_EXECUTABLE_REGULAR_FILE: '+str(p))
    # Preserve argv[0] basename (cargo/rustc); rustup dispatches on it.
    return p

def trusted_regular_node(path):
    """Resolve Node ONLY for strict Rust PluginSandbox::new path validation.

    Cargo MUST retain its original symlink/argv0 because rustup dispatches by name.
    Node needs a non-symlink, absolute, executable regular file under .voder-tools.
    """
    original = checked_toolchain_executable(path)
    resolved = original.resolve(strict=True)
    if not resolved.is_absolute() or resolved.is_symlink():
        raise RuntimeError('RUST_NODE_PATH_NOT_PLAIN: ' + str(original))
    if not stat.S_ISREG(resolved.lstat().st_mode) or not os.access(resolved, os.X_OK):
        raise RuntimeError('RUST_NODE_PATH_NOT_EXECUTABLE_REGULAR_FILE: ' + str(resolved))
    return resolved

def rust_binary(path):
    p=checked(path)
    if not os.access(p,os.X_OK) or p.open('rb').read(4) != b'\x7fELF':
        raise RuntimeError('EXPECTED_REAL_ELF_RUST_BINARY: '+str(p))
    return p

def owned(path,parent):
    p=Path(path).resolve(strict=True); root=Path(parent).resolve(strict=True)
    if root not in p.parents: raise RuntimeError('OUTSIDE_SCOPE: '+str(p))
    return p

def run(argv, *, input=None, timeout=30, env=None):
    try:
        return subprocess.run([str(x) for x in argv],input=input,text=True,stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE,timeout=timeout,env=env,check=False)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f'TIMED_OUT_{timeout}s: {argv[0]}') from e

def atomic_report(path,report):
    path=Path(path); tmp=path.with_name('.'+path.name+'.tmp')
    fd=os.open(tmp,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'w') as f:
        json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def events(journal):
    out=[]
    for line in checked(journal).read_text(encoding='utf-8').splitlines():
        row=json.loads(line)
        e=row.get('body')
        if not isinstance(e,dict): raise RuntimeError('BAD_JOURNAL_EVENT')
        out.append(e)
    return out

def files_digest_tree(base):
    base=checked(base,'dir')
    data=[]
    for p in sorted(base.rglob('*')):
        if p.is_symlink(): raise RuntimeError('SYMLINK_IN_SCOPE')
        if p.is_file():
            data.append((str(p.relative_to(base)), digest(p.read_bytes())))
        elif not p.is_dir(): raise RuntimeError('SPECIAL_FILE_IN_SCOPE')
    return digest(json.dumps(data,sort_keys=True).encode())
