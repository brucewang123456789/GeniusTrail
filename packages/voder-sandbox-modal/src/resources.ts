import {readFileSync,statfsSync,writeFileSync,renameSync} from 'node:fs';
import {freemem,totalmem,tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {randomUUID} from 'node:crypto';
import {Worker} from 'node:worker_threads';
import {getHeapStatistics} from 'node:v8';
const MiB=1024*1024;
export type ResourceSample={availableMemory:number;memoryCapacity:number;memoryReserve:number;diskAvailable:number;diskReserve:number;rss:number};
function numberFile(path:string):number|undefined{try{const n=Number(readFileSync(path,'utf8').trim());return Number.isFinite(n)&&n>0&&n<Number.MAX_SAFE_INTEGER?n:undefined;}catch{return undefined;}}
export function resourceSample(diskPath=tmpdir()):ResourceSample{
 let available=freemem(),capacity=totalmem();
 try{const match=/^MemAvailable:\s+(\d+) kB$/m.exec(readFileSync('/proc/meminfo','utf8'));if(match)available=Number(match[1])*1024;}catch{}
 // Inspect this process's cgroup and all mounted ancestors, including a shared
 // container limit. Host free memory alone is insufficient inside containers.
 try{
  const row=readFileSync('/proc/self/cgroup','utf8').split('\n').find(s=>s.startsWith('0::'));
  const relative=row?.slice(3)??'/';let group=resolve('/sys/fs/cgroup','.'+relative);
  if(!group.startsWith('/sys/fs/cgroup/'))group='/sys/fs/cgroup';
  for(;;){
   const limit=numberFile(join(group,'memory.max')),used=numberFile(join(group,'memory.current'));
   if(limit!==undefined&&used!==undefined){capacity=Math.min(capacity,limit);available=Math.min(available,Math.max(0,limit-used));}
   if(group==='/sys/fs/cgroup')break;group=resolve(group,'..');if(!group.startsWith('/sys/fs/cgroup'))break;
  }
  const v1limit=numberFile('/sys/fs/cgroup/memory/memory.limit_in_bytes'),v1used=numberFile('/sys/fs/cgroup/memory/memory.usage_in_bytes');
  if(v1limit!==undefined&&v1used!==undefined){capacity=Math.min(capacity,v1limit);available=Math.min(available,Math.max(0,v1limit-v1used));}
 }catch{}
 const disk=statfsSync(diskPath,{bigint:true});const diskTotal=Number(disk.blocks*disk.bsize);
 return {availableMemory:available,memoryCapacity:capacity,memoryReserve:Math.max(64*MiB,Math.min(512*MiB,capacity*.1)),diskAvailable:Number(disk.bavail*disk.bsize),diskReserve:Math.max(64*MiB,Math.min(1024*MiB,diskTotal*.05)),rss:process.memoryUsage.rss()};
}
export function pressureReason(s:ResourceSample,pendingMemory=0,pendingDisk=0):string|undefined{
 if(s.availableMemory-pendingMemory<s.memoryReserve)return 'MEMORY_HEADROOM_LOW';
 if(s.diskAvailable-pendingDisk<s.diskReserve)return 'DISK_HEADROOM_LOW';
 return undefined;
}
export function assertAllocation(bytes:number):void{
 const s=resourceSample();const reason=pressureReason(s,bytes*2);
 if(!Number.isSafeInteger(bytes)||bytes<0||reason)throw Error('RESOURCE_PRESSURE:'+String(reason??'UNREPRESENTABLE_SIZE'));
}
export class ResourceGuard{
 readonly flag=new Int32Array(new SharedArrayBuffer(4));
 readonly evidence=join(tmpdir(),'voder-resource-stop-'+randomUUID()+'.json');
 readonly controller=new AbortController();
 private worker:Worker|undefined;private closed=false;private problem:Error|undefined;
 private sampledAt=0;private sampled:ResourceSample|undefined;private heapLow=false;
 readonly diskPath:string;
 constructor(diskPath=tmpdir()){
  this.diskPath=diskPath;this.check();
  this.worker=new Worker(new URL('./resource-worker.mjs',import.meta.url),{workerData:{flag:this.flag.buffer,evidence:this.evidence,diskPath},execArgv:['--experimental-strip-types']});
  this.worker.unref();
  this.worker.on('message',(record)=>this.trip(String(record.reason)));
  this.worker.on('error',()=>this.trip('RESOURCE_MONITOR_UNAVAILABLE'));
 }
 private trip(reason:string):never|void{
  if(this.problem)return;
  Atomics.store(this.flag,0,1);this.problem=Error('RESOURCE_PRESSURE:'+reason+'; PRIVATE_EVIDENCE='+this.evidence);
  try{writeFileSync(this.evidence,JSON.stringify({schema:'voder.resource-stop.v1',reason,time_utc:new Date().toISOString(),sample:resourceSample(this.diskPath)})+'\n',{mode:0o600,flag:'wx'});}catch{}
  this.controller.abort(this.problem);
 }
 check(pendingMemory=0,pendingDisk=0):void{
  if(this.closed)throw Error('RESOURCE_GUARD_CLOSED');
  if(Atomics.load(this.flag,0))this.trip('INDEPENDENT_MONITOR_STOP');
  if(this.problem)throw this.problem;
  if(!this.sampled||Date.now()-this.sampledAt>=100){this.sampled=resourceSample(this.diskPath);this.sampledAt=Date.now();const heap=getHeapStatistics();this.heapLow=heap.used_heap_size>heap.heap_size_limit*.8;}
  const reason=pressureReason(this.sampled,pendingMemory,pendingDisk)??(this.heapLow?'JS_HEAP_HEADROOM_LOW':undefined);
  if(reason){this.trip(reason);throw this.problem;}
 }
 checkMetadata(bytes:number):void{
  this.check(bytes);const heap=getHeapStatistics();if(heap.used_heap_size+bytes>heap.heap_size_limit*.8){this.trip('JS_HEAP_HEADROOM_LOW');throw this.problem;}
 }
 async run<T>(work:()=>Promise<T>):Promise<T>{
  this.check();const signal=this.controller.signal;if(signal.aborted)throw signal.reason;
  let listener:()=>void=()=>{};
  const stop=new Promise<never>((_,reject)=>{listener=()=>reject(signal.reason);signal.addEventListener('abort',listener,{once:true});});
  try{return await Promise.race([work(),stop]);}finally{signal.removeEventListener('abort',listener);}
 }
 async close():Promise<void>{if(this.closed)return;this.closed=true;await this.worker?.terminate();}
}
