#!/usr/bin/env python3
import importlib.util, pathlib, tempfile, os, stat, subprocess
from pathlib import Path
pkg=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('voder_common',pkg/'common.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with tempfile.TemporaryDirectory() as td:
    root=Path(td);tools=root/'.voder-tools';toolbin=tools/'rust/cargo/bin';toolbin.mkdir(parents=True)
    rustup=toolbin/'rustup';rustup.write_text('#!/bin/sh\ncase "${0##*/}" in cargo) echo CARGO_DISPATCH_OK;; *) echo BAD_DISPATCH; exit 9;; esac\n');rustup.chmod(0o755)
    cargo=toolbin/'cargo';cargo.symlink_to('rustup')
    old=m.ROOT;m.ROOT=root
    assert m.checked_toolchain_executable(cargo)==cargo, 'CLI name lost after proxy validation'
    out=subprocess.run([str(m.checked_toolchain_executable(cargo)),'build','--locked'],capture_output=True,text=True)
    assert out.returncode==0 and out.stdout.strip()=='CARGO_DISPATCH_OK',str(out)
    try:
        m.checked_toolchain_executable(rustup.parent/'../../../../escape')
    except Exception:pass
    else:raise AssertionError('escape not blocked')
    outside=root/'outside';outside.write_text('#!/bin/sh\n');outside.chmod(0o755)
    bad=toolbin/'bad';bad.symlink_to(outside)
    try:m.checked_toolchain_executable(bad)
    except RuntimeError as e:assert 'TOOLCHAIN_OUTSIDE_PINNED_ROOT' in str(e)
    else:raise AssertionError('out-of-tree link accepted')
    m.ROOT=old
print('CARGO_RUSTUP_PROXY_ARGV0_REGRESSION=PASS')
