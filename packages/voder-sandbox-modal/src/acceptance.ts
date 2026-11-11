/** Live (billable) Modal sandbox acceptance: positive and negative cases.
 * Invocation needs configured MODAL_TOKEN_ID/SECRET, app and digest-pinned image.
 * This test exercises the real Modal SDK and cloud compute, not fake drivers.
 */
import {mkdtemp,mkdir,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {createModalDriver} from './driver.ts';
import {executeModalTask} from './execute.ts';
const h=(data:string)=>createHash('sha256').update(data).digest('hex');
async function main(){
  const work=await mkdtemp(join(tmpdir(),'voder-modal-acceptance-'));
  try{
    await mkdir(join(work,'src'));
    await writeFile(join(work,'src','check.txt'),'VODER_MODAL_ACCEPTANCE_OK\n');
    const backend=await createModalDriver();
    const task={workspace:work,program:'/usr/bin/cat',args:['src/check.txt'],
      runId:'vgr5-live',actionId:'positive-1',requiredFiles:['src/check.txt'],
      timeoutMs:90_000,memoryMiB:512,cpu:1,maxOutputBytes:16_384};
    const positive=await executeModalTask(task,backend);
    if(positive.status!=='EXECUTED'||positive.exitCode!==0||positive.timeout||positive.outputLimited||
       positive.stdout!=='VODER_MODAL_ACCEPTANCE_OK\n'||positive.stdoutSha256!==h(positive.stdout)||
       positive.stderrSha256!==h(positive.stderr)||positive.runId!==task.runId||positive.actionId!==task.actionId){
      throw new Error('MODAL_SUCCESS_RECEIPT_INVALID');
    }
    console.log('MODAL_VALIDATED_SUCCESS=PASS');
    const negative=await executeModalTask({...task,program:'/usr/bin/false',args:[],actionId:'negative-1'},backend);
    if(negative.status!=='EXECUTED'||negative.exitCode===0||negative.timeout||negative.outputLimited||
       negative.stdoutSha256!==h(negative.stdout)||negative.stderrSha256!==h(negative.stderr)){
      throw new Error('MODAL_NONZERO_EXIT_RECEIPT_INVALID');
    }
    console.log('MODAL_VALIDATED_NONZERO_EXIT=PASS');
    // Invalid inputs must be rejected locally and must not reach Modal.
    let blocked=false;
    try{
      await executeModalTask({...task,actionId:'missing-1',requiredFiles:['src/no-such-file.txt']},backend);
    }catch {blocked=true;}
    if(!blocked)throw new Error('MISSING_INPUT_GATE_BYPASSED');
    console.log('MODAL_MISSING_INPUT_FAIL_CLOSED=PASS');
    console.log('MODAL_LIVE_ACCEPTANCE=PASS');
  }finally{
    await rm(work,{recursive:true,force:true});
  }
}
main().catch(e=>{
  console.error(`MODAL_LIVE_ACCEPTANCE=FAIL:${e instanceof Error?e.message:String(e)}`);
  process.exitCode=1;
});
