/**
 * Voder-owned read-only repository capabilities.
 * No exec, write, filesystem-following link, hidden file or host command access.
 * Trusted native VGR is the only mutation authority.
 */
import { constants } from 'node:fs';
import {createHash} from 'node:crypto';
import { lstat, open, readlink, readdir, realpath } from 'node:fs/promises';
import { isAbsolute, join, relative, resolve, sep } from 'node:path';
import { VoderError } from './contracts.mjs';

// A page bounds model context, never the size of the underlying file.
const PAGE_BYTES = 4096;
const MAX_LIST_ENTRIES = 150;
const MAX_SEARCH_FILES = 400;
const MAX_SEARCH_BYTES = 2 * 1024 * 1024;
const MAX_SEARCH_MATCHES = 40;
const DENY = new Set(['.git','.svn','.hg','.env','.ssh','.aws','.config','node_modules','venv','.venv','__pycache__','.voder']);
function validatePath(path, allowRoot = false) {
  if (allowRoot && (path === '' || path === '.')) return [];
  if (typeof path !== 'string' || !path || path.length > 400 || path.includes('\0') || path.includes('\\') || isAbsolute(path)) {
    throw new VoderError('INVALID_PATH', 'Invalid repository-relative path');
  }
  const parts = path.split('/');
  if (parts.some(p => !p || p === '.' || p === '..' || p.startsWith('.') || DENY.has(p))) {
    throw new VoderError('INVALID_PATH', 'Traversal, hidden entries and metadata paths are forbidden');
  }
  return parts;
}
function within(base, file) {
  const rel = relative(base,file);
  return rel === '' || (rel !== '..' && !rel.startsWith('..'+sep) && !isAbsolute(rel));
}
export class RepositoryInspector {
  constructor(cwd) { if (typeof cwd !== 'string' || !isAbsolute(cwd)) throw new VoderError('INVALID_WORKSPACE','Absolute workspace required'); this.cwd = resolve(cwd); }
  async base() { return realpath(this.cwd); }
  async safePath(path, { allowRoot=false, kind='file' }={}) {
    const base = await this.base();
    const parts = validatePath(path, allowRoot);
    let cursor = base;
    for (const p of parts) {
      cursor = join(cursor,p);
      const meta = await lstat(cursor);
      if (meta.isSymbolicLink()) throw new VoderError('UNTRUSTED_PATH','Symlinks forbidden in repository tool paths');
    }
    const canonical = await realpath(cursor);
    if (!within(base,canonical)) throw new VoderError('UNTRUSTED_PATH','Escaped repository');
    const meta = await lstat(canonical);
    if ((kind==='file' && !meta.isFile()) || (kind==='dir' && !meta.isDirectory()) ||
        (kind==='search' && !meta.isFile() && !meta.isDirectory())) throw new VoderError('INVALID_PATH',`Expected repository ${kind==='search'?'file or directory':kind}; inspect the parent directory to choose a valid path`);
    return { base, full:canonical,meta };
  }
  async _stream(path, signal, consume) {
    if (signal?.aborted) throw new VoderError('CANCELLED','Read cancelled');
    const {base,full} = await this.safePath(path);
    const handle = await open(full,constants.O_RDONLY|constants.O_NOFOLLOW|constants.O_NONBLOCK);
    try {
      const before = await handle.stat({bigint:true});
      if (!before.isFile()) throw new VoderError('UNTRUSTED_PATH','Unsafe file descriptor');
      const actual = await readlink(`/proc/self/fd/${handle.fd}`);
      if (!within(base,actual) || actual.endsWith(' (deleted)')) throw new VoderError('UNTRUSTED_PATH','Opened inode escaped repository');
      const hash=createHash('sha256'),decoder=new TextDecoder('utf-8',{fatal:true,ignoreBOM:true});
      const buffer=Buffer.alloc(64*1024);let position=0;
      for(;;){
        if(signal?.aborted)throw new VoderError('CANCELLED','Read cancelled');
        const remaining=Number(before.size)-position;if(remaining<=0)break;
        const {bytesRead}=await handle.read(buffer,0,Math.min(buffer.length,remaining),null);if(!bytesRead)break;
        const bytes=buffer.subarray(0,bytesRead);if(bytes.includes(0))throw new VoderError('NOT_TEXT','Binary file');
        let text;try{text=decoder.decode(bytes,{stream:true});}catch{throw new VoderError('NOT_TEXT','File is not valid UTF-8 text');}
        hash.update(bytes);consume?.(bytes,text,position);position+=bytesRead;
      }
      try{const tail=decoder.decode();if(tail)consume?.(Buffer.alloc(0),tail,position);}catch{throw new VoderError('NOT_TEXT','File is not valid UTF-8 text');}
      const after=await handle.stat({bigint:true});
      const current=await lstat(full,{bigint:true});
      if(['dev','ino','size','mtimeNs','ctimeNs'].some(k=>before[k]!==after[k]||after[k]!==current[k])||BigInt(position)!==after.size)
        throw new VoderError('SOURCE_CHANGED','File changed while reading; reread its current hash');
      return {path,bytes:position,sha256:hash.digest('hex')};
    } finally { await handle.close(); }
  }
  // Trusted edit expansion only; never registered as a model read tool.
  async readFull({path},signal){
    const chunks=[];const file=await this._stream(path,signal,bytes=>chunks.push(Buffer.from(bytes)));
    return {...file,content:Buffer.concat(chunks).toString('utf8'),partial:false,eof:true};
  }
  async read({path,offset=0,limit=PAGE_BYTES},signal){
    if(!Number.isSafeInteger(offset)||offset<0||!Number.isSafeInteger(limit)||limit<4||limit>16384)
      throw new VoderError('INVALID_BUDGET','offset is a nonnegative UTF-8 byte position; limit is a page size of 4..16384 bytes');
    const chunks=[];
    const file=await this._stream(path,signal,(bytes,_text,start)=>{
      const lo=Math.max(0,offset-start),hi=Math.min(bytes.length,offset+limit+4-start);
      if(hi>lo)chunks.push(Buffer.from(bytes.subarray(lo,hi)));
    });
    const page=Buffer.concat(chunks);let end=Math.min(limit,page.length);
    if(page.length&&(page[0]&0xc0)===0x80)throw new VoderError('INVALID_OFFSET','offset must lie on a UTF-8 boundary; use next_offset');
    while(end>0&&end<page.length&&(page[end]&0xc0)===0x80)end--;
    const content=page.subarray(0,end).toString('utf8'),next=offset+end,eof=next>=file.bytes;
    return {...file,content,offset,returned_bytes:end,partial:offset>0||!eof,eof,next_offset:eof?null:next,
      ...(offset>0||!eof?{notice:'PAGE ONLY. sha256 and bytes describe the WHOLE file; use next_offset to continue. Prefer pinned exact edits for modifications.'}:{})};
  }
  async list({path=''}, signal) {
    if (signal?.aborted) throw new VoderError('CANCELLED','List cancelled');
    const {full} = await this.safePath(path,{allowRoot:true,kind:'dir'});
    const rows = await readdir(full,{withFileTypes:true});
    if (rows.length>MAX_LIST_ENTRIES) throw new VoderError('LIST_TOO_LARGE','Directory entry budget exceeded');
    return {path,entries:rows.filter(x=>!x.name.startsWith('.')&&!DENY.has(x.name)&&!x.isSymbolicLink())
      .map(x=>({name:x.name,kind:x.isDirectory()?'directory':x.isFile()?'file':'other'}))
      .sort((a,b)=>a.name.localeCompare(b.name))};
  }
  async search({query,max_results=20,path=''},signal) {
    if (typeof query!=='string'||!query.trim()||query.length>160||query.includes('\0')) throw new VoderError('INVALID_QUERY','Search needs a bounded literal query');
    if (!Number.isSafeInteger(max_results)||max_results<1||max_results>MAX_SEARCH_MATCHES) throw new VoderError('INVALID_BUDGET','Search matches must be 1..40');
    const {base,meta} = await this.safePath(path,{allowRoot:true,kind:'search'});
    const pending=meta.isDirectory()?[path==='.'?'':path]:[];let files=0,scanned=0;const matches=[];
    const partial=reason=>({query,path,matches,truncated:true,complete:false,limit_reason:reason,files_scanned:files,bytes_scanned:scanned,next_step:'Narrow path to a source file or directory; incomplete search does not prove absence.'});
    const scan=async filePath=>{
      let line=1,carry='',found=false;
      const file=await this._stream(filePath,signal,(_bytes,text)=>{
        // Retain only enough overlap for a literal match crossing stream chunks.
        const parts=text.split('\n');for(let i=0;i<parts.length;i++){const part=parts[i];
          const joined=carry+part,at=joined.indexOf(query);
          if(!found&&at>=0&&matches.length<max_results){matches.push({path:filePath,line,text:joined.slice(Math.max(0,at-60),Math.max(0,at-60)+240)});found=true;}
          carry=query.length>1?joined.slice(-(query.length-1)):'';
          // The newline is handled below, including the final empty segment.
          if(i<parts.length-1){line++;carry='';found=false;}
        }
      });
      scanned+=file.bytes;return matches.length>=max_results;
    };
    // A literal file path is a first-class search scope, using the SAME safe reader.
    // Explicit text files are streamed regardless of their total byte length.
    if(meta.isFile()){files=1;if(await scan(path))return partial('MATCHES');}
    while(pending.length){
      if(signal?.aborted)throw new VoderError('CANCELLED','Search cancelled');
      const dir=pending.shift();
      const entries=await readdir(join(base,dir),{withFileTypes:true});
      if(entries.length>MAX_LIST_ENTRIES)return partial('DIRECTORY_ENTRIES');
      for(const e of entries){
        if(e.name.startsWith('.')||DENY.has(e.name)||e.isSymbolicLink())continue;
        const path=dir?`${dir}/${e.name}`:e.name;
        if(e.isDirectory()){
          if(pending.length>80)return partial('DIRECTORIES');
          pending.push(path);
        } else if(e.isFile()){
          if(files>=MAX_SEARCH_FILES)return partial('FILES');files++;
          const st=await lstat(join(base,path));
          if(scanned>=MAX_SEARCH_BYTES)return {...partial('BYTES'),next_file:path};
          try{if(await scan(path))return partial('MATCHES');}catch(e){
            if(e.code==='SEARCH_BYTE_BUDGET')return partial('BYTES');
            if(['NOT_TEXT','UNTRUSTED_PATH','FILE_TOO_LARGE','ENOENT'].includes(e.code)){scanned+=st.size;continue;}
            throw e;
          }
        }
      }
    }
    return {query,path,matches,truncated:false,complete:true,files_scanned:files,bytes_scanned:scanned};
  }
}

/** Agent-side tool routing only. Never grant permissions through tool descriptions. */
export function repositoryReadTools(cwd){
  const inspector=new RepositoryInspector(cwd);
  return [
    {name:'voder_read_file',description:'Read a UTF-8 source page; no whole-file size ceiling. offset is a byte position; follow next_offset. sha256 always covers the entire file. No links or host files.',
      parameters:{type:'object',properties:{path:{type:'string'},offset:{type:'integer',minimum:0,description:'UTF-8 byte offset; start at 0 or use next_offset'},limit:{type:'integer',minimum:4,maximum:16384,description:'Bytes per response page; default 4096, not a file-size limit'}},required:['path'],additionalProperties:false},execute:(args,signal)=>inspector.read(args,signal)},
    {name:'voder_list_files',description:'List a repository directory. Use path empty string or dot for root. No links or hidden entries.',
      parameters:{type:'object',properties:{path:{type:'string'}},required:['path'],additionalProperties:false},execute:(args,signal)=>inspector.list(args,signal)},
    {name:'voder_search_code',description:'Find literal text in one repository file OR recursively in a directory (max 40 matches). Example: {query:"strconv",path:"project/builtin/lib.go"}. Omit path or use empty string for repository root. Prefer a relevant file/directory; no globs, links, hidden or host files. If complete=false, narrow path; incomplete search never proves absence.',
      parameters:{type:'object',properties:{query:{type:'string',description:'Literal text, not a regular expression',minLength:1,maxLength:160},path:{type:'string',description:'Repository-relative file or directory; empty string or dot means root',maxLength:400},max_results:{type:'integer',minimum:1,maximum:40}},required:['query'],additionalProperties:false},execute:(args,signal)=>inspector.search(args,signal)}
  ];
}
