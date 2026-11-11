#!/usr/bin/env python3
"""Positive/negative regression for Rust Node vs Cargo path semantics."""
import importlib.util, os, stat, subprocess, tempfile
from pathlib import Path
spec=importlib.util.spec_from_file_location('voder_t08_common',Path(__file__).with_name('common.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with tempfile.TemporaryDirectory() as temp:
    r=Path(temp);tool_root=r/'.voder-tools'; nbin=tool_root/'node/bin'; real=tool_root/'node-versions/node-v24.13.0-linux-x64/bin/node'
    nbin.mkdir(parents=True);real.parent.mkdir(parents=True)
    real.write_bytes(b'#!/bin/sh\necho NODE_REAL_OK\n');real.chmod(0o755)
    link=nbin/'node';link.symlink_to(real)
    old=m.ROOT;m.ROOT=r
    try:
        assert m.checked_toolchain_executable(link)==link
        resolved=m.trusted_regular_node(link)
        assert resolved==real and not resolved.is_symlink() and stat.S_ISREG(resolved.lstat().st_mode)
        p=subprocess.run([str(resolved)],capture_output=True,text=True)
        assert p.returncode==0 and p.stdout.strip()=='NODE_REAL_OK'
        print('TEST_RUST_NODE_SYMLINK_RESOLUTION=PASS')
        cargo_root=tool_root/'rust/cargo/bin';cargo_root.mkdir(parents=True)
        rustup=cargo_root/'rustup';rustup.write_text('#!/bin/sh\ncase "${0##*/}" in cargo) echo CARGO_OK ;; *) exit 9 ;; esac\n');rustup.chmod(0o755)
        cargo=cargo_root/'cargo';cargo.symlink_to('rustup')
        assert m.checked_toolchain_executable(cargo)==cargo
        assert subprocess.run([str(cargo)],capture_output=True,text=True).stdout.strip()=='CARGO_OK'
        print('TEST_CARGO_PROXY_ARGV0_UNCHANGED=PASS')
        outside=r/'outside';outside.write_text('#!/bin/sh\n');outside.chmod(0o755)
        bad=nbin/'malicious';bad.symlink_to(outside)
        try: m.trusted_regular_node(bad)
        except RuntimeError as e: assert 'TOOLCHAIN_OUTSIDE_PINNED_ROOT' in str(e)
        else: raise AssertionError('outside toolchain accepted')
        print('TEST_NODE_OUTSIDE_TOOLCHAIN_REJECTED=PASS')
        noexec=nbin/'noexec';noexec.write_text('noop\n')
        try: m.trusted_regular_node(noexec)
        except RuntimeError as e: assert 'TOOLCHAIN_NOT_EXECUTABLE_REGULAR_FILE' in str(e)
        else: raise AssertionError('non-executable node accepted')
        print('TEST_NODE_NONEXECUTABLE_REJECTED=PASS')
        try: m.trusted_regular_node(nbin/'missing')
        except Exception: pass
        else: raise AssertionError('missing node accepted')
        print('TEST_NODE_MISSING_REJECTED=PASS')
    finally:
        m.ROOT=old
print('T08_NODE_REALPATH_REGRESSION=PASS')
