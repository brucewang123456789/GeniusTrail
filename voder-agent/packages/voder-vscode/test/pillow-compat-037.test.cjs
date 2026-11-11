'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),path=require('node:path');
const {spawnSync}=require('node:child_process');
const {decoderPython}=require('../src/document-input.cjs');
const fixtures=process.env.VODER_ARCHIVE_FIXTURES||path.resolve(__dirname,'../../../../fixtures036');
test('legacy Pillow API without Image.Resampling decodes and resizes the full reported-size image',()=>{
 const script=String.raw`
import contextlib,io,json,os,sys,types,importlib.util
import PIL
from PIL import Image as real,ImageOps
real.init()
data=open(sys.argv[2],'rb').read()
# Expose the real Pillow codec through the public API of old Pillow releases.
# Real pixel decoding/resizing/encoding still executes; no image function is mocked.
legacy=types.SimpleNamespace(open=real.open,LANCZOS=getattr(real,'Resampling',real).LANCZOS,MAX_IMAGE_PIXELS=real.MAX_IMAGE_PIXELS)
PIL.Image=legacy
os.environ['VODER_INPUT_MEMORY_BYTES']=str(8*1024**3)
spec=importlib.util.spec_from_file_location('decoder_compat',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
sys.argv=['decoder','image','.png']
sys.stdin=types.SimpleNamespace(buffer=io.BytesIO(data))
out=io.BytesIO();sys.stdout=types.SimpleNamespace(buffer=out)
try:module.main()
finally:sys.stdout=sys.__stdout__;PIL.Image=real
result=json.loads(out.getvalue())
assert result['originalSize']==[12387,7498]
assert result['outputSize']==[8192,4959]
assert not hasattr(legacy,'Resampling')
print('LEGACY_PILLOW_FULL_IMAGE_RESIZE=PASS')
`;
 const r=spawnSync(decoderPython(),['-I','-B','-c',script,path.resolve(__dirname,'../src/document-convert.py'),path.join(fixtures,'large-12387x7498.png')],{encoding:'utf8',timeout:120000,maxBuffer:8192});
 assert.equal(r.status,0,r.stderr);assert.match(r.stdout,/LEGACY_PILLOW_FULL_IMAGE_RESIZE=PASS/);
});
