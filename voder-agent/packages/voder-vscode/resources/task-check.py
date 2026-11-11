"""Host-pinned task checks. User commands run only in the independent sandbox."""
import ast
import hashlib
import math
import json
import re
import os
import signal
import resource
import shutil
import subprocess
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

CONFIG = None  # Replaced with the reviewed, host-validated scope.

def check():
    before = {}
    for relative in CONFIG['files']:
        p = Path(relative)
        if p.is_symlink() or not p.is_file():
            raise ValueError('missing/nonregular target: ' + relative)
        raw = p.read_bytes()
        if not raw or b'\x00' in raw:
            raise ValueError('invalid text candidate: ' + relative)
        text = raw.decode('utf-8', errors='strict')
        before[relative] = hashlib.sha256(raw).hexdigest()
        suffix = p.suffix.lower()
        if suffix == '.py':
            ast.parse(text, filename=relative)
        elif suffix == '.json':
            json.loads(text)
        elif suffix in ('.svg', '.xml'):
            if re.search(r'<!\s*(?:DOCTYPE|ENTITY)|<\?xml-stylesheet', text, re.I):
                raise ValueError('XML external declarations forbidden')
            tree = ET.fromstring(text)
            if suffix == '.svg':
                if tree.tag.split('}')[-1] != 'svg':
                    raise ValueError('SVG root missing')
                if 'viewBox' in tree.attrib:
                    box = [float(x) for x in re.split(r'[ ,]+', tree.attrib['viewBox'].strip())]
                    if len(box) != 4 or not all(math.isfinite(x) for x in box) or min(box[2:]) <= 0:
                        raise ValueError('invalid SVG viewBox')
                ids = set()
                refs = []
                for e in tree.iter():
                    if e.tag.split('}')[-1].lower() in ('script', 'foreignobject', 'image', 'feimage', 'iframe', 'audio', 'video'):
                        raise ValueError('SVG executable or raster content forbidden')
                    if e.tag.split('}')[-1].lower() == 'style':
                        css = ''.join(e.itertext())
                        if re.search(r'@import|https?:|data:|file:', css, re.I):
                            raise ValueError('external SVG stylesheet forbidden')
                        for link in re.findall(r'url\(\s*[\'\"]?([^\)\'\"]+)', css):
                            if not link.startswith('#'):
                                raise ValueError('external SVG stylesheet URL forbidden')
                            refs.append(link[1:])
                    for key, value in e.attrib.items():
                        name = key.split('}')[-1].lower()
                        if name == 'id':
                            if value in ids:
                                raise ValueError('duplicate SVG id: ' + value)
                            ids.add(value)
                        if name.startswith('on'):
                            raise ValueError('SVG event handler forbidden')
                        if name in ('href', 'src'):
                            if not value.startswith('#'):
                                raise ValueError('external SVG reference forbidden')
                            refs.append(value[1:])
                        for link in re.findall(r'url\(\s*[\'\"]?([^\)\'\"]+)', value):
                            if not link.startswith('#'):
                                raise ValueError('external SVG URL forbidden')
                            refs.append(link[1:])
                if any(ref not in ids for ref in refs):
                    raise ValueError('unresolved SVG reference')
        print(json.dumps({'file': relative, 'sha256': before[relative], 'check': 'text/syntax', 'visual_check': 'NOT_RUN'}))
    for argv in CONFIG['checks']:
        if not shutil.which(argv[0]):
            raise RuntimeError('TASK_TOOL_UNAVAILABLE: ' + argv[0])
        # Output goes to a temporary file, never an unbounded in-memory pipe.
        with tempfile.TemporaryFile() as output:
            def limits():
                resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))
            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=output, stderr=output, start_new_session=True, preexec_fn=limits)
            try:
                code = child.wait(timeout=20)
            finally:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
            result = type('CommandResult', (), {'returncode': code})()
            output.seek(0, 2)
            size = output.tell()
            output.seek(max(0, size - 8000))
            tail = output.read(8000).decode('utf-8', errors='replace')
        print(json.dumps({'argv': argv, 'exit_code': result.returncode, 'output_tail': tail, 'output_truncated': size > 8000}))
        if result.returncode:
            raise RuntimeError('TASK_CHECK_FAILED: ' + repr(argv))
    for name, digest in before.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError('check modified candidate source: ' + name)
    print('TASK_CONFIGURED_CHECKS_PASS; VISUAL_QUALITY_NOT_CERTIFIED')

if __name__ == '__main__':
    check()
