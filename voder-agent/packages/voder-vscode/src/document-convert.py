"""Read-only file decoder. Input/output bytes use stdio; no user code is executed."""
import base64, io, json, os, pathlib, re, shutil, subprocess, sys, tempfile, zipfile
import xml.etree.ElementTree as ET

# Bound native codec/parser allocations to current host resources, not a fixed file quota.
try:
    import resource
    memory=int(os.environ.get('VODER_INPUT_MEMORY_BYTES','0'))
    if memory:
        soft,hard=resource.getrlimit(resource.RLIMIT_AS)
        bound=memory*3//4
        if hard!=resource.RLIM_INFINITY:bound=min(bound,hard)
        resource.setrlimit(resource.RLIMIT_AS,(bound,hard))
except (ImportError,ValueError,OSError):pass

MAX_INPUT=int(os.environ.get('VODER_INPUT_FILE_BYTES',str(128*1024*1024)))
MAX_EXPANDED=int(os.environ.get('VODER_INPUT_EXPANDED_BYTES',str(256*1024*1024)))

class DecodeError(Exception): pass

def checked_zip(data):
    try:z=zipfile.ZipFile(io.BytesIO(data))
    except Exception:raise DecodeError('REFERENCE_ARCHIVE_INVALID')
    infos=z.infolist()
    if sum(x.file_size+len(x.filename)+256 for x in infos)>MAX_EXPANDED:raise DecodeError('REFERENCE_ARCHIVE_LIMIT')
    for x in infos:
        p=pathlib.PurePosixPath(x.filename)
        if p.is_absolute() or '..' in p.parts or '\\' in x.filename or x.flag_bits&1:raise DecodeError('REFERENCE_ARCHIVE_UNSAFE')
        if x.file_size>MAX_EXPANDED:raise DecodeError('REFERENCE_ARCHIVE_LIMIT')
    for member in z.namelist():
        if member.endswith('.rels'):
            for rel in ET.fromstring(z.read(member)):
                if rel.get('TargetMode')=='External' and not str(rel.get('Type','')).endswith('/hyperlink'):
                    raise DecodeError('REFERENCE_EXTERNAL_RESOURCE')
    return z

def spreadsheet_text(z):
    ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    relns='{http://schemas.openxmlformats.org/package/2006/relationships}'
    links={x.get('Id'):x.get('Target') for x in ET.fromstring(z.read('xl/_rels/workbook.xml.rels')) if x.tag==relns+'Relationship' and x.get('TargetMode')!='External'}
    shared=[]
    if 'xl/sharedStrings.xml' in z.namelist():
        shared=[''.join(x.itertext()) for x in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si',ns)]
    lines=['[SPREADSHEET CELLS: all stored sheets/rows; formulas are not executed; cached results may be stale]']
    for sheet in ET.fromstring(z.read('xl/workbook.xml')).findall('m:sheets/m:sheet',ns):
        target=links.get(sheet.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'))
        if not target:continue
        member=target.lstrip('/') if target.startswith('/') else 'xl/'+target
        if '..' in pathlib.PurePosixPath(member).parts:raise DecodeError('REFERENCE_ARCHIVE_UNSAFE')
        lines.append('[SHEET '+str(sheet.get('name'))+']')
        tree=ET.fromstring(z.read(member))
        for row in tree.findall('m:sheetData/m:row',ns):
            cells=[]
            for c in row.findall('m:c',ns):
                v=c.find('m:v',ns);f=c.find('m:f',ns);value=v.text if v is not None else ''
                if c.get('t')=='s' and value is not None:value=shared[int(value)]
                if c.get('t')=='inlineStr':value=''.join(c.find('m:is',ns).itertext())
                cells.append(str(c.get('r'))+'='+json.dumps({'formula':f.text if f is not None else None,'stored_value':value},ensure_ascii=False))
            lines.append('\t'.join(cells))
    return '\n'.join(lines)

def convert(data,ext):
    # A private LibreOffice profile disables macros, link updates, and document events.
    program=shutil.which('libreoffice') or shutil.which('soffice')
    if not program:raise DecodeError('REFERENCE_OFFICE_DEPENDENCY')
    with tempfile.TemporaryDirectory(prefix='voder-document-') as tmp:
        p=pathlib.Path(tmp);profile=p/'profile';profile.mkdir()
        (profile/'user').mkdir()
        (profile/'user/registrymodifications.xcu').write_text('''<?xml version="1.0"?><oor:items xmlns:oor="http://openoffice.org/2001/registry"><item oor:path="/org.openoffice.Office.Common/Security/Scripting"><prop oor:name="MacroSecurityLevel" oor:op="fuse"><value>3</value></prop><prop oor:name="DisableMacrosExecution" oor:op="fuse"><value>true</value></prop></item><item oor:path="/org.openoffice.Office.Common/Security/Scripting"><prop oor:name="SecureURL" oor:op="fuse"><value/></prop></item><item oor:path="/org.openoffice.Office.Writer/Content/Update"><prop oor:name="Link" oor:op="fuse"><value>0</value></prop></item><item oor:path="/org.openoffice.Office.Calc/Content/Update"><prop oor:name="Link" oor:op="fuse"><value>0</value></prop></item></oor:items>''')
        src=p/('input'+ext);src.write_bytes(data)
        result=subprocess.run([program,'-env:UserInstallation='+profile.as_uri(),'--headless','--nologo','--nodefault','--norestore','--nolockcheck','--convert-to','pdf','--outdir',str(p),str(src)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=100,env={**os.environ,'HOME':tmp,'SAL_USE_VCLPLUGIN':'svp'})
        dest=p/'input.pdf'
        if result.returncode or not dest.is_file():raise DecodeError('REFERENCE_OFFICE_CONVERSION')
        if dest.stat().st_size>MAX_INPUT:raise DecodeError('REFERENCE_LIMIT')
        return dest.read_bytes()

def main():
    op,ext=sys.argv[1:3];data=sys.stdin.buffer.read(MAX_INPUT+1)
    if len(data)>MAX_INPUT:raise DecodeError('REFERENCE_FILE_LIMIT')
    if op in ('image','image4096','verify'):
        from PIL import Image, ImageOps
        memory=int(os.environ.get('VODER_INPUT_MEMORY_BYTES',str(1024**3)))
        # Decode capacity follows free/cgroup memory. A second allocation check is
        # done before pixels are loaded; Pillow's historical 40MP cutoff is gone.
        pixel_budget=max(1,memory//24)
        Image.MAX_IMAGE_PIXELS=pixel_budget
        im=Image.open(io.BytesIO(data))
        if getattr(im,'n_frames',1)>1:raise DecodeError('REFERENCE_ANIMATED_IMAGE')
        if im.width*im.height>pixel_budget:raise DecodeError('REFERENCE_IMAGE_RESOURCE')
        original_size=list(im.size)
        fmt=im.format
        # Pillow 9.0.x TIFF getexif() requires the open stream; load() closes it.
        rotate=im.getexif().get(274,1)!=1
        if fmt=='TIFF':
            # The TIFF loader already applies Orientation; do not transpose twice.
            im.load()
            im=im.copy()
            im.info.pop('exif',None)
        else:
            if rotate:im=ImageOps.exif_transpose(im)
            im.load()
        if op=='verify':
            result={'decoded':True,'width':im.width,'height':im.height}
        else:
            # The official DeepSeek endpoint accepts at most 8192 on a side;
            # reduced copies retain the complete canvas and declare the transform.
            side=4096 if op=='image4096' else 8192
            resized=max(im.size)>side
            if fmt in ('PNG','JPEG','WEBP') and not resized and not rotate:
                result={'decoded':True,'unchanged':True,'originalSize':original_size,'outputSize':original_size}
            else:
                im.thumbnail((side,side),getattr(Image,'Resampling',Image).LANCZOS)
                out=io.BytesIO();im.convert('RGBA').save(out,format='PNG')
                result={'image':base64.b64encode(out.getvalue()).decode(),'mimeType':'image/png','originalSize':original_size,'outputSize':list(im.size),'resized':resized,'orientationCorrected':rotate}
    elif op=='pickle':
        # Pickle is accepted as data, never deserialized: GLOBAL/REDUCE may execute code.
        import pickletools
        lines=['[PICKLE STATIC OPCODES: no unpickling, imports, constructors or reductions executed]']
        used=0
        for opcode,arg,position in pickletools.genops(data):
            line=str(position)+' '+opcode.name+' '+repr(arg)
            used+=len(line.encode('utf-8'))
            if used>MAX_EXPANDED:raise DecodeError('REFERENCE_PICKLE_RESOURCE')
            lines.append(line)
        result={'text':'\n'.join(lines)}
    elif op=='office':
        z=checked_zip(data) if data.startswith(b'PK') else None
        text=spreadsheet_text(z) if z and 'xl/workbook.xml' in z.namelist() else ''
        if ext in ('.xls','.xlt','.xla','.xlw') and not z:
            import xlrd
            book=xlrd.open_workbook(file_contents=data,on_demand=True)
            blocks=['[LEGACY SPREADSHEET: all stored rows; values are stored results, formulas are not recalculated]']
            for sheet in book.sheets():
                blocks.append('[SHEET '+sheet.name+']')
                for i in range(sheet.nrows):
                    blocks.append(str(i+1)+'\t'+json.dumps(sheet.row_values(i),ensure_ascii=False))
            text='\n'.join(blocks)
        # Apple packages carrying a PDF preview can be faithfully read without executing content.
        pdf=None
        if ext in ('.pages','.key','.keynote','.numbers') and z:
            pdf=next((z.read(n) for n in ('QuickLook/Preview.pdf','preview.pdf') if n in z.namelist()),None)
        if pdf is None:pdf=convert(data,ext)
        result={'pdf':base64.b64encode(pdf).decode(),'structuredText':text}
    else:raise DecodeError('REFERENCE_FORMAT')
    encoded=json.dumps(result,ensure_ascii=False).encode()
    if len(encoded)>MAX_INPUT*2:raise DecodeError('REFERENCE_LIMIT')
    sys.stdout.buffer.write(encoded)

if __name__=='__main__':
    try:main()
    except Exception as e:
        # No document content, paths, or parser diagnostics in public errors.
        code=str(e) if isinstance(e,DecodeError) else 'REFERENCE_IMAGE_RESOURCE' if type(e).__name__ in ('MemoryError','DecompressionBombError') else 'REFERENCE_DOCUMENT_DECODE'
        print(json.dumps({'error':code,'class':type(e).__name__}))
