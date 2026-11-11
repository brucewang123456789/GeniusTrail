import {lstat,opendir,open,realpath} from 'node:fs/promises';
import {constants as fsConstants,openSync,fstatSync,readFileSync,closeSync} from 'node:fs';
import {join,relative,sep} from 'node:path';
import {createHash} from 'node:crypto';
import {assertRelative} from './contract.ts';
import {ResourceGuard,assertAllocation} from './resources.ts';
export type InputFile={relative:string;readonly bytes:Uint8Array;sha256:string;source:string;size:number;ino:number;dev:number;mtimeMs:number};
export type InputSnapshot={files:InputFile[];digest:string;totalBytes:number;readonly checksumText:string};
const SKIP_DIR=new Set(['.git','.venv','venv','node_modules','target','.next','.cache','.idea','__pycache__']);
const DENY_NAME=/(^|\/)(\.env(?:\..*)?|\.npmrc|\.pypirc|\.modal\.toml|id_(?:rsa|ed25519)|credentials(?:\.json)?|\.ssh|\.aws|\.kube)(\/|$)/i;
const hash=(bytes:Uint8Array|string)=>createHash('sha256').update(bytes).digest('hex');
function same(info:{ino:number;dev:number;size:number;mtimeMs:number;nlink:number;isFile():boolean},file:InputFile):boolean{
 return info.isFile()&&info.nlink===1&&info.ino===file.ino&&info.dev===file.dev&&info.size===file.size&&info.mtimeMs===file.mtimeMs;
}
/** Stream a pinned file and detect any mismatch before an archive is released. */
export async function* fileChunks(file:InputFile,guard:ResourceGuard):AsyncGenerator<Buffer>{
 guard.check(2*1024*1024);const handle=await open(file.source,fsConstants.O_RDONLY|fsConstants.O_NOFOLLOW);const digest=createHash('sha256');let size=0;
 try{
  if(!same(await handle.stat(),file))throw Error('WORKSPACE_CHANGED_DURING_SNAPSHOT');
  for(;;){guard.check(2*1024*1024);const buffer=Buffer.allocUnsafe(1024*1024);const result=await handle.read(buffer,0,buffer.length,null);if(!result.bytesRead)break;
   const bytes=buffer.subarray(0,result.bytesRead);size+=bytes.length;digest.update(bytes);yield bytes;
  }
  if(!same(await handle.stat(),file)||size!==file.size||digest.digest('hex')!==file.sha256)throw Error('WORKSPACE_CHANGED_DURING_SNAPSHOT');
 }finally{await handle.close();}
}
function materialize(file:InputFile):Uint8Array{
 // Compatibility for existing explicit bytes consumers; production snapshot,
 // archive and upload never use this getter. Reserve for copies before allocating.
 assertAllocation(file.size);const fd=openSync(file.source,fsConstants.O_RDONLY|fsConstants.O_NOFOLLOW);
 try{if(!same(fstatSync(fd),file))throw Error('WORKSPACE_CHANGED_DURING_SNAPSHOT');const bytes=readFileSync(fd);
  if(!same(fstatSync(fd),file)||bytes.length!==file.size||hash(bytes)!==file.sha256)throw Error('WORKSPACE_CHANGED_DURING_SNAPSHOT');return bytes;
 }finally{closeSync(fd);}
}
export async function snapshotWorkspace(root:string,requiredFiles:string[]=[]):Promise<InputSnapshot>{
 const guard=new ResourceGuard();
 try{
  const rootStat=await lstat(root);if(!rootStat.isDirectory()||rootStat.isSymbolicLink())throw Error('INVALID_WORKSPACE');
  const top=await realpath(root);const stat=await lstat(top);if(!stat.isDirectory()||stat.isSymbolicLink())throw Error('INVALID_WORKSPACE');
  const files:InputFile[]=[];let totalBytes=0,checksumBytes=0;const digest=createHash('sha256');
  async function walk(dir:string):Promise<void>{
   guard.check();const entries=[];const directory=await opendir(dir);
   for await(const entry of directory){guard.checkMetadata(2048);entries.push(entry);}
   guard.checkMetadata(entries.length*64);entries.sort((a,b)=>a.name.localeCompare(b.name,'en'));
   for(const entry of entries){guard.check(2*1024*1024);const abs=join(dir,entry.name),rel=assertRelative(relative(top,abs).split(sep).join('/'));
    if(entry.isSymbolicLink())throw Error('SYMLINK_BLOCKED:'+rel);
    if(entry.name==='.git'&&entry.isFile())continue;if(DENY_NAME.test(rel))continue;
    if(entry.isDirectory()){if(!SKIP_DIR.has(entry.name))await walk(abs);continue;}
    if(!entry.isFile())throw Error('SPECIAL_FILE_BLOCKED:'+rel);
    const info=await lstat(abs);if(info.nlink!==1)throw Error('HARDLINK_BLOCKED:'+rel);
    if(!Number.isSafeInteger(info.size)||!Number.isSafeInteger(totalBytes+info.size))throw Error('INPUT_SIZE_UNREPRESENTABLE');
    const handle=await open(abs,fsConstants.O_RDONLY|fsConstants.O_NOFOLLOW);const itemHash=createHash('sha256');let size=0;
    const file={relative:rel,source:abs,size:info.size,ino:info.ino,dev:info.dev,mtimeMs:info.mtimeMs,sha256:''} as InputFile;
    try{
     if(!same(await handle.stat(),file))throw Error('WORKSPACE_CHANGED_DURING_SNAPSHOT');
     const buf=Buffer.allocUnsafe(Math.min(1024*1024,Math.max(1,info.size)));
     for(;;){guard.check(2*1024*1024);const read=await handle.read(buf,0,buf.length,null);if(!read.bytesRead)break;size+=read.bytesRead;itemHash.update(buf.subarray(0,read.bytesRead));}
     if(!same(await handle.stat(),file)||size!==info.size)throw Error('WORKSPACE_CHANGED_DURING_SNAPSHOT');
    }finally{await handle.close();}
    file.sha256=itemHash.digest('hex');Object.defineProperty(file,'bytes',{get:()=>materialize(file),enumerable:false});
    files.push(file);totalBytes+=size;const line=file.sha256+'  '+rel+'\n';digest.update(line);checksumBytes+=Buffer.byteLength(line);
   }
  }
  await walk(top);guard.checkMetadata(files.length*128);const names=new Set(files.map(f=>f.relative));for(const name of requiredFiles)if(!names.has(assertRelative(name)))throw Error('MISSING_REQUIRED_FILE:'+name);
  if(!files.length){digest.update('\n');checksumBytes=1;}
  return {files,totalBytes,digest:digest.digest('hex'),get checksumText(){assertAllocation(checksumBytes*3);return files.map(f=>f.sha256+'  '+f.relative).join('\n')+'\n';}};
 }finally{await guard.close();}
}
