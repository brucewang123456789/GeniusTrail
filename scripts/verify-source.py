#!/usr/bin/env python3
"""Verify byte-for-byte completeness of this review candidate, no writes."""
from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parent.parent
manifest=json.loads((root/'SOURCE_MANIFEST_SHA256.json').read_text())
expected = manifest['files']
optional = manifest.get('optional_generated_archive_files', {})
ignored_dirs = {'.git', '.voder-build', '.voder-tools', 'node_modules', 'dist', 'target', '__pycache__'}
actual = {}
for file in root.rglob('*'):
    relative = file.relative_to(root)
    name = relative.as_posix()
    if name == 'SOURCE_MANIFEST_SHA256.json' or not file.is_file():
        continue
    if name not in expected and (name in optional or ignored_dirs.intersection(relative.parts)):
        continue
    if name not in expected and name in {
        'voder-agent/packages/voder-search/native/bin/voder-retrieval',
        'voder-agent/packages/voder-vscode/src/vendor/playwright-core-1.61.1.tgz',
    }:
        continue
    actual[name] = hashlib.sha256(file.read_bytes()).hexdigest()
missing = sorted(expected.keys() - actual.keys())
extra = sorted(actual.keys() - expected.keys())
changed = sorted(p for p in expected.keys() & actual.keys() if actual[p] != expected[p])
print(f'REQUIRED_FILES={len(expected)} MISSING={len(missing)} EXTRA={len(extra)} CHANGED={len(changed)}')
for label, files in [('MISSING', missing), ('EXTRA', extra), ('CHANGED', changed)]:
    for name in files[:15]:
        print(label, name)
if missing or extra or changed:
    sys.exit(2)
print('RESULT=SHA256_SOURCE_INTEGRITY_PASS')
