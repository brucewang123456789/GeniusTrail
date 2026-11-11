'use strict';
/** Capacity is a resource budget, not a file-extension/count quota. */
const fs=require('node:fs'),os=require('node:os'),v8=require('node:v8'),{constants}=require('node:buffer');
function availableMemory(){
 let available=os.freemem();
 try{const m=fs.readFileSync('/proc/meminfo','utf8').match(/^MemAvailable:\s+(\d+) kB/m);if(m)available=Number(m[1])*1024;}catch{}
 for(const [max,current] of [['/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current'],['/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/memory/memory.usage_in_bytes']]){
  try{const limit=Number(fs.readFileSync(max,'utf8')),used=Number(fs.readFileSync(current,'utf8'));if(Number.isFinite(limit)&&limit>0&&Number.isFinite(used))available=Math.min(available,Math.max(0,limit-used));}catch{}
 }
 return Math.floor(Math.max(0,available));
}
// Leave room for base64, JSON serialization, parser buffers, and the running host.
const memory=availableMemory(),heap=v8.getHeapStatistics().heap_size_limit;
const wireBytes=Math.floor(Math.min(memory/8,heap/8,constants.MAX_STRING_LENGTH/2,constants.MAX_LENGTH/8));
const LIMITS=Object.freeze({items:Math.floor(wireBytes/1024),blocks:Math.floor(wireBytes/1024),textBytes:Math.floor(wireBytes/2),imageBytes:Math.floor(wireBytes/2),wireBytes,fileBytes:Math.floor(Math.min(memory/8,constants.MAX_LENGTH/4)),expandedBytes:Math.floor(Math.min(memory/8,heap/8,constants.MAX_STRING_LENGTH/2)),memoryBytes:memory});
module.exports={availableMemory,LIMITS};
