/** Bounded-memory gzip stream; independent byte verification remains mandatory. */
import {createHash} from 'node:crypto';
import {createGzip} from 'node:zlib';
import {Readable,Transform} from 'node:stream';
import {pipeline} from 'node:stream/promises';
import {createReadStream,createWriteStream} from 'node:fs';
import {mkdtemp,chmod,rm,readFile,stat} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {assertRelative} from './contract.ts';
import {fileChunks,type InputSnapshot,type InputFile} from './manifest.ts';
import {ResourceGuard} from './resources.ts';
const hash=(b:Uint8Array|string)=>createHash('sha256').update(b).digest('hex');
const cache=new Map<string,Buffer>();const CACHE_BYTES=32*1024*1024;let cachedBytes=0;
export type InputArchive={path:string;sha256:string;size:number;cacheHit:boolean;dispose():Promise<void>};
function header(name:string,size:number,type='0'):Buffer{
 const b=Buffer.alloc(512);b.write(name,0,100,'utf8');const oct=(n:number,w:number)=>n.toString(8).padStart(w-1,'0')+'\0';
 b.write(oct(0o600,8),100,8);b.write(oct(0,8),108,8);b.write(oct(0,8),116,8);
 // PAX size records handle files larger than the traditional 8 GiB tar field.
 b.write(oct(size>0o77777777777?0:size,12),124,12);b.write(oct(0,12),136,12);b.fill(32,148,156);
 b.write(type,156,1);b.write('ustar\0',257,6);b.write('00',263,2);let sum=0;for(const byte of b)sum+=byte;b.write(sum.toString(8).padStart(6,'0')+'\0 ',148,8);return b;
}
function pax(key:string,value:string):Buffer{
 const tail=' '+key+'='+value+'\n';let size=Buffer.byteLength(tail)+1;
 for(;;){const next=Buffer.byteLength(tail)+String(size).length;if(next===size)break;size=next;}return Buffer.from(String(size)+tail);
}
async function* smallEntry(name:string,bytes:Buffer,type='0'):AsyncGenerator<Buffer>{yield header(name,bytes.length,type);yield bytes;if(bytes.length%512)yield Buffer.alloc(512-bytes.length%512);}
function manifestSize(snapshot:InputSnapshot):number{return snapshot.files.reduce((sum,f)=>sum+Buffer.byteLength(f.sha256+'  '+f.relative+'\n'),0)||(snapshot.files.length?0:1);}
async function* entries(snapshot:InputSnapshot,guard:ResourceGuard,cacheOnly=false):AsyncGenerator<Buffer>{
 const digest=createHash('sha256');const names=new Set<string>();let total=0;
 for(const [i,file] of snapshot.files.entries()){
  guard.check(4*1024*1024);assertRelative(file.relative);
  if(names.has(file.relative)||file.relative==='.voder-input.sha256')throw Error('INPUT_ARCHIVE_BINDING_MISMATCH');names.add(file.relative);
  digest.update(file.sha256+'  '+file.relative+'\n');total+=file.size;
  const metadata=Buffer.concat([pax('path',file.relative),pax('size',String(file.size))]);
  if(!cacheOnly){yield* smallEntry('PaxHeaders/'+i,metadata,'x');yield header('input-'+i,file.size);}
  for await(const bytes of fileChunks(file,guard)){if(!cacheOnly)yield bytes;}
  if(!cacheOnly&&file.size%512)yield Buffer.alloc(512-file.size%512);
 }
 if(!snapshot.files.length)digest.update('\n');
 if(total!==snapshot.totalBytes||digest.digest('hex')!==snapshot.digest)throw Error('INPUT_ARCHIVE_BINDING_MISMATCH');
 if(cacheOnly)return;
 const size=manifestSize(snapshot);yield header('.voder-input.sha256',size);
 if(snapshot.files.length){for(const file of snapshot.files){guard.check();yield Buffer.from(file.sha256+'  '+file.relative+'\n');}}else yield Buffer.from('\n');
 if(size%512)yield Buffer.alloc(512-size%512);yield Buffer.alloc(1024);
}
export async function inputArchive(snapshot:InputSnapshot,externalGuard?:ResourceGuard):Promise<InputArchive>{
 const guard=externalGuard??new ResourceGuard();let directory:string|undefined;let output:ReturnType<typeof createWriteStream>|undefined;let source:Readable|undefined;
 try{
  guard.check();directory=await mkdtemp(join(tmpdir(),'voder-input-stream-'));await chmod(directory,0o700);
  const path=join(directory,'input.tar.gz');let size=0;const digest=createHash('sha256');const saved=cache.get(snapshot.digest);
  if(saved){for await(const _ of entries(snapshot,guard,true)){} }
  source=saved?Readable.from([saved]):Readable.from(entries(snapshot,guard),{objectMode:false,highWaterMark:1024*1024});
  const meter=new Transform({transform(chunk:Buffer,_encoding,callback){try{guard.check(4*1024*1024,chunk.length);digest.update(chunk);size+=chunk.length;callback(null,chunk);}catch(e){callback(e as Error);}}});
  output=createWriteStream(path,{flags:'wx',mode:0o600});output.on('error',()=>{});
  if(saved)await guard.run(()=>pipeline(source!,meter,output!,{signal:guard.controller.signal}));
  else await guard.run(()=>pipeline(source!,createGzip({level:1,chunkSize:64*1024}),meter,output!,{signal:guard.controller.signal}));
  await chmod(path,0o400);const resultHash=digest.digest('hex');
  if(!saved&&size<=CACHE_BYTES){guard.check(size*3);const bytes=await readFile(path);
   while(cache.size&&cachedBytes+bytes.length>CACHE_BYTES){const key=cache.keys().next().value!;cachedBytes-=cache.get(key)!.length;cache.delete(key);}cache.set(snapshot.digest,bytes);cachedBytes+=bytes.length;
  }
  const owned=directory;return {path,sha256:resultHash,size,cacheHit:!!saved,dispose:()=>rm(owned,{recursive:true,force:true})};
 }catch(error){source?.destroy();output?.destroy();if(directory)await rm(directory,{recursive:true,force:true});throw error;}
 finally{if(!externalGuard)await guard.close();}
}
/** SDK only supports writeBytes: bounded chunks, not one aggregate Buffer. */
export async function uploadArchive(archive:InputArchive,filesystem:{writeBytes(bytes:Uint8Array,path:string):Promise<void>},guard:ResourceGuard):Promise<number>{
 const digest=createHash('sha256');let bytes=0,count=0;
 const source=createReadStream(archive.path,{highWaterMark:8*1024*1024});
 try{
  for await(const value of source){guard.check(32*1024*1024);const chunk=value as Buffer;digest.update(chunk);bytes+=chunk.length;
   const path='/tmp/voder-input.part-'+String(count++).padStart(16,'0');await guard.run(()=>filesystem.writeBytes(chunk,path));
  }
  if(bytes!==archive.size||digest.digest('hex')!==archive.sha256)throw Error('LOCAL_ARCHIVE_CHANGED_DURING_UPLOAD');
  return count;
 }finally{source.destroy();}
}
