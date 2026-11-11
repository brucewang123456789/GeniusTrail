/** Independent sampler: shared stop flag remains observable even if main JS is busy. */
import {parentPort,workerData} from 'node:worker_threads';
import {readFileSync,writeFileSync,renameSync} from 'node:fs';
import {resourceSample,pressureReason} from './resources.ts';
const stopped=new Int32Array(workerData.flag);
const sample=()=>{
 try{
  const value=resourceSample(workerData.diskPath);const reason=pressureReason(value);
  if(!reason)return;
  const record={schema:'voder.resource-stop.v1',reason,time_utc:new Date().toISOString(),sample:value,monitor:'INDEPENDENT_WORKER'};
  const pending=workerData.evidence+'.pending';writeFileSync(pending,JSON.stringify(record)+'\n',{mode:0o600,flag:'wx'});renameSync(pending,workerData.evidence);
  Atomics.store(stopped,0,1);parentPort.postMessage(record);clearInterval(timer);
 }catch(error){
  Atomics.store(stopped,0,1);parentPort.postMessage({reason:'RESOURCE_MONITOR_UNAVAILABLE',error:String(error)});clearInterval(timer);
 }
};
const timer=setInterval(sample,100);sample();
