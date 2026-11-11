"""Read libarchive formats in memory; never extract paths or run archive programs."""
import base64, ctypes as C, ctypes.util, hashlib, json, os, pathlib, posixpath, re, sys

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

class ArchiveError(Exception): pass
def require(ok, code):
    if not ok: raise ArchiveError(code)

def read_archive(data, suffix, scan=False, destination=None):
    name=ctypes.util.find_library('archive')
    require(name, 'REFERENCE_ARCHIVE_DEPENDENCY')
    lib=C.CDLL(name)
    def fn(name, restype, *args):
        f=getattr(lib,name); f.restype=restype; f.argtypes=list(args); return f
    ptr=C.c_void_p; integer=C.c_int; size=C.c_size_t
    new=fn('archive_read_new',ptr)
    filters=fn('archive_read_support_filter_all',integer,ptr)
    formats=fn('archive_read_support_format_all',integer,ptr)
    raw=fn('archive_read_support_format_raw',integer,ptr)
    opened=fn('archive_read_open_memory',integer,ptr,ptr,size)
    header=fn('archive_read_next_header',integer,ptr,C.POINTER(ptr))
    pathname=fn('archive_entry_pathname_utf8',C.c_char_p,ptr)
    kind=fn('archive_entry_filetype',C.c_uint,ptr)
    entrysize=fn('archive_entry_size',C.c_longlong,ptr)
    symlink=fn('archive_entry_symlink',C.c_char_p,ptr)
    hardlink=fn('archive_entry_hardlink',C.c_char_p,ptr)
    encrypted=fn('archive_entry_is_encrypted',integer,ptr)
    read=fn('archive_read_data',C.c_ssize_t,ptr,ptr,size)
    free=fn('archive_read_free',integer,ptr)
    fmt=fn('archive_format_name',C.c_char_p,ptr)
    filtername=fn('archive_filter_name',C.c_char_p,ptr,integer)
    budget=int(os.environ['VODER_INPUT_EXPANDED_BYTES'])
    a=new(); require(a,'REFERENCE_ARCHIVE_RESOURCE')
    buffer=C.create_string_buffer(data); total=0; entries=[]; paths=set(); types={}; descendants=set(); metadata=0; format_name=''
    try:
        filters(a); formats(a)
        # Raw mode is used only for declared standalone compression streams.
        if suffix in ('.gz','.bz2','.xz','.zst','.zstd','.lz4','.lzma','.lz','.lzo','.z','.br'):
            raw(a)
        require(opened(a,buffer,len(data))==0,'REFERENCE_ARCHIVE_INVALID_OR_UNSUPPORTED')
        entry=ptr(); chunk=C.create_string_buffer(1024*1024)
        while True:
            rc=header(a,C.byref(entry))
            if rc==1: break
            require(rc==0,'REFERENCE_ARCHIVE_CORRUPT_ENCRYPTED_OR_UNSUPPORTED')
            format_name=(fmt(a) or b'unknown').decode('utf-8','replace')
            encoded=pathname(entry)
            require(encoded is not None,'REFERENCE_ARCHIVE_PATH_ENCODING')
            try: p=encoded.decode('utf-8','strict')
            except UnicodeError: raise ArchiveError('REFERENCE_ARCHIVE_PATH_ENCODING')
            if format_name=='raw': p=re.sub(r'\.(?:gz|bz2|xz|zst|zstd|lz4|lzma|lz|lzo|z|br)$','',sys.argv[4],flags=re.I) if len(sys.argv)>4 else 'content'
            require(p and not p.startswith(('/','\\')) and not re.match(r'^[A-Za-z]:',p) and '\\' not in p and not any(ord(x)<32 or ord(x)==127 for x in p),'REFERENCE_ARCHIVE_PATH')
            # A harmless leading ./ is common in tar; all traversal is rejected.
            while p.startswith('./'): p=p[2:]
            p=p.rstrip('/')
            if p in ('','.') and kind(entry)==0o040000: continue
            require(p and all(x not in ('','..','.') for x in p.split('/')),'REFERENCE_ARCHIVE_PATH')
            require(len(p.encode())<=1024,'REFERENCE_ARCHIVE_PATH_TOO_LONG')
            require(not symlink(entry) and not hardlink(entry),'REFERENCE_ARCHIVE_LINK')
            require(encrypted(entry)==0,'REFERENCE_ARCHIVE_ENCRYPTED')
            t=kind(entry)
            require(t in (0o100000,0o040000,0),'REFERENCE_ARCHIVE_SPECIAL_FILE')
            require(p not in paths,'REFERENCE_ARCHIVE_DUPLICATE_PATH')
            parents=[]; q=posixpath.dirname(p)
            while q: parents.append(q); q=posixpath.dirname(q)
            require(not any(types.get(q)=='file' for q in parents),'REFERENCE_ARCHIVE_PATH_CONFLICT')
            isdir=t==0o040000
            require(isdir or p not in descendants,'REFERENCE_ARCHIVE_PATH_CONFLICT')
            descendants.update(parents)
            paths.add(p); types[p]='directory' if isdir else 'file'
            require(entrysize(entry)<=budget-total,'REFERENCE_ARCHIVE_RESOURCE')
            parts=[]; n=0; hasher=hashlib.sha256(); output=None
            if destination and not isdir:
                dest=pathlib.Path(destination).joinpath(*p.split('/'))
                dest.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
                output=os.open(dest,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            elif destination and isdir:
                pathlib.Path(destination).joinpath(*p.split('/')).mkdir(parents=True,exist_ok=True,mode=0o700)
            while True:
                count=read(a,chunk,len(chunk))
                require(count>=0,'REFERENCE_ARCHIVE_CORRUPT_ENCRYPTED_OR_UNSUPPORTED')
                if not count: break
                n+=count; total+=count
                require(total<=budget,'REFERENCE_ARCHIVE_RESOURCE')
                block=chunk.raw[:count];hasher.update(block)
                if output is not None:
                    with os.fdopen(os.dup(output),'wb') as f:f.write(block)
                elif not scan:parts.append(block)
            if output is not None:os.close(output)
            content=b''.join(parts)
            require(not isdir or not content,'REFERENCE_ARCHIVE_PATH_CONFLICT')
            e={'path':p,'kind':'directory' if isdir else 'file','bytes':n,'sha256':hasher.hexdigest()}
            if not isdir and not scan: e['data']=base64.b64encode(content).decode('ascii')
            entries.append(e)
            metadata+=len(p.encode())+256
            require(metadata+total<=budget and metadata<=int(os.environ['VODER_INPUT_MEMORY_BYTES'])//32,'REFERENCE_ARCHIVE_RESOURCE')
        format_name=format_name or (fmt(a) or b'').decode('utf-8','replace')
        require(bool(format_name),'REFERENCE_ARCHIVE_EMPTY_OR_UNSUPPORTED')
        # A raw reader must have actually used a compression filter, not accepted arbitrary bytes.
        if format_name=='raw': require((filtername(a,0) or b'none')!=b'none','REFERENCE_ARCHIVE_UNSUPPORTED_COMPRESSION')
        return {'format':format_name,'entries':entries,'expandedBytes':total}
    finally: free(a)

if __name__=='__main__':
    try:
        limit=int(os.environ['VODER_INPUT_FILE_BYTES']); data=sys.stdin.buffer.read(limit+1)
        require(len(data)<=limit,'REFERENCE_ARCHIVE_RESOURCE')
        scan=sys.argv[1] in ('archive-scan','archive-extract')
        destination=sys.argv[3] if sys.argv[1]=='archive-extract' else None
        if destination:
            st=os.lstat(destination)
            require(__import__('stat').S_ISDIR(st.st_mode) and not st.st_mode&0o077 and not any(pathlib.Path(destination).iterdir()),'REFERENCE_ARCHIVE_DESTINATION')
        result=read_archive(data,sys.argv[2].lower(),scan=scan,destination=destination)
        print(json.dumps(result,ensure_ascii=True,separators=(',',':')))
    except Exception as e:
        print(json.dumps({'error':str(e) if isinstance(e,ArchiveError) else 'REFERENCE_ARCHIVE_RESOURCE' if isinstance(e,MemoryError) else 'REFERENCE_ARCHIVE_DECODE','class':type(e).__name__}))
