'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),path=require('node:path');
const {spawnSync}=require('node:child_process');
const {decoderPython}=require('../src/document-input.cjs');
const fixtures=process.env.VODER_DOCUMENT_FIXTURES||path.resolve(__dirname,'../../../../public/Voder_File_Multimodal_0.1.35_Patch/fixtures');
test('TIFF metadata is read before pixel loading closes the stream and is not reparsed afterward',()=>{
 const script=String.raw`
import io,json,os,sys,types,importlib.util
from PIL import Image
real_open=Image.open
seen=[]
def open_legacy(*a,**kw):
    im=real_open(*a,**kw)
    if im.format=='TIFF':
        getexif=im.getexif;load=im.load;state={'cached':False,'loaded':False}
        def legacy_getexif():
            if state['loaded'] and not state['cached']:
                raise AttributeError("'NoneType' object has no attribute 'seek'")
            value=getexif();state['cached']=True;seen.append('exif');return value
        def legacy_load(*a,**kw):
            state['loaded']=True;seen.append('load');return load(*a,**kw)
        im.getexif=legacy_getexif;im.load=legacy_load
    return im
Image.open=open_legacy
os.environ['VODER_INPUT_MEMORY_BYTES']=str(8*1024**3)
spec=importlib.util.spec_from_file_location('decoder',sys.argv[1]);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
data=open(sys.argv[2],'rb').read();sys.argv=['decoder','image','.tiff']
sys.stdin=types.SimpleNamespace(buffer=io.BytesIO(data));out=io.BytesIO();sys.stdout=types.SimpleNamespace(buffer=out)
try:module.main()
finally:sys.stdout=sys.__stdout__;Image.open=real_open
result=json.loads(out.getvalue());assert result['mimeType']=='image/png';assert seen.index('exif')<seen.index('load'),seen
png=real_open(io.BytesIO(__import__('base64').b64decode(result['image'])));png.load()
assert png.size==tuple(result['outputSize']);print('TIFF_EXIF_BEFORE_LOAD=PASS')
`;
 const r=spawnSync(decoderPython(),['-I','-B','-c',script,path.resolve(__dirname,'../src/document-convert.py'),path.join(fixtures,'sample.tiff')],{encoding:'utf8',timeout:30000,maxBuffer:8192});
 assert.equal(r.status,0,r.stderr);assert.match(r.stdout,/TIFF_EXIF_BEFORE_LOAD=PASS/);
});
test('all eight TIFF orientations across raw/LZW/deflate and PNG EXIF retain exact correctly oriented pixels',()=>{
 const script=String.raw`
import base64,io,json,os,sys,types,importlib.util
from PIL import Image
os.environ['VODER_INPUT_MEMORY_BYTES']=str(8*1024**3)
spec=importlib.util.spec_from_file_location('decoder_orientation',sys.argv[1]);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
base=Image.new('RGB',(7,5));base.putdata([(x*30,y*40,(x+y)*20) for y in range(5) for x in range(7)])
T=getattr(Image,'Transpose',Image)
methods={2:T.FLIP_LEFT_RIGHT,3:T.ROTATE_180,4:T.FLIP_TOP_BOTTOM,5:T.TRANSPOSE,6:T.ROTATE_270,7:T.TRANSVERSE,8:T.ROTATE_90}
count=0
for fmt,compression in [('TIFF','raw'),('TIFF','tiff_lzw'),('TIFF','tiff_adobe_deflate'),('PNG',None)]:
    for orientation in range(1,9):
        exif=Image.Exif();exif[274]=orientation
        src=io.BytesIO();opts={'exif':exif}
        if compression:opts['compression']=compression
        base.save(src,format=fmt,**opts);data=src.getvalue()
        sys.argv=['decoder','image','.tiff' if fmt=='TIFF' else '.png']
        sys.stdin=types.SimpleNamespace(buffer=io.BytesIO(data));out=io.BytesIO();saved=sys.stdout;sys.stdout=types.SimpleNamespace(buffer=out)
        try:module.main()
        finally:sys.stdout=saved
        r=json.loads(out.getvalue());payload=base64.b64decode(r['image']) if 'image' in r else data
        actual=Image.open(io.BytesIO(payload)).convert('RGB')
        expected=base.transpose(methods[orientation]) if orientation in methods else base.copy()
        assert actual.size==expected.size,(fmt,compression,orientation,actual.size,expected.size)
        assert actual.tobytes()==expected.tobytes(),(fmt,compression,orientation,'pixels')
        assert Image.open(io.BytesIO(payload)).getexif().get(274,1)==1,(fmt,compression,orientation,'stale orientation')
        count+=1
print('ORIENTATION_PIXEL_CASES='+str(count)+'; PASS=True')
`;
 const r=spawnSync(decoderPython(),['-I','-B','-c',script,path.resolve(__dirname,'../src/document-convert.py')],{encoding:'utf8',timeout:30000,maxBuffer:8192});
 assert.equal(r.status,0,r.stderr);assert.match(r.stdout,/ORIENTATION_PIXEL_CASES=32; PASS=True/);
});
