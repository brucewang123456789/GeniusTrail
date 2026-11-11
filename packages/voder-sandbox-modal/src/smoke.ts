/** A live, billable authenticated Modal test. Never invoked in offline preflight. */
import {mkdtemp,mkdir,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createModalDriver} from './driver.ts';
import {executeModalTask} from './execute.ts';
async function main(){
  const work=await mkdtemp(join(tmpdir(),'voder-modal-smoke-'));
  try{
    await mkdir(join(work,'src'));
    await writeFile(join(work,'src','check.txt'),'VODER_MODAL_SMOKE_OK\n');
    const receipt=await executeModalTask({workspace:work,program:'/usr/bin/cat',args:['src/check.txt'],
      runId:'smoke',actionId:'smoke-1',requiredFiles:['src/check.txt'],
      timeoutMs:60_000,memoryMiB:512,cpu:1,maxOutputBytes:8192},await createModalDriver());
    if(receipt.exitCode!==0||receipt.stdout!=='VODER_MODAL_SMOKE_OK\n'||receipt.outputLimited)throw new Error('LIVE_RECEIPT_GATE_FAILED');
    console.log('MODAL_REMOTE_LIVE_GATE=PASS');
    console.log('MODAL_SANDBOX_ID='+receipt.sandboxId);
  }finally{await rm(work,{recursive:true,force:true});}
}
main().catch(e=>{console.error('MODAL_REMOTE_LIVE_GATE=BLOCKED '+String(e));process.exitCode=1;});
