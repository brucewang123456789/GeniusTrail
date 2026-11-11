/** Provider-agnostic transport receipt conformance (not a VGR commit certificate). */
import {createHash} from 'node:crypto';
import type {Receipt,SandboxTask} from './contract.ts';
const HEX=/^[0-9a-f]{64}$/;
const digest=(s:string)=>createHash('sha256').update(s).digest('hex');
export function assertProviderReceipt(record:Receipt,task:SandboxTask,provider:string):Receipt{
  if(!record||typeof record!=='object'||record.protocol!=='voder-sandbox-receipt-v1'||record.provider!==provider||
    record.runId!==task.runId||record.actionId!==task.actionId||record.status!=='EXECUTED'||
    typeof record.sandboxId!=='string'||!record.sandboxId||record.sandboxId.length>256||
    typeof record.image!=='string'||!record.image||record.image.length>512||
    typeof record.inputDigest!=='string'||!HEX.test(record.inputDigest)||
    !Number.isSafeInteger(record.files)||record.files<1||
    !Number.isSafeInteger(record.bytes)||record.bytes<1||
    typeof record.exitCode!=='number'||!Number.isSafeInteger(record.exitCode)||record.exitCode<0||record.exitCode>255||
    typeof record.timeout!=='boolean'||typeof record.outputLimited!=='boolean'||
    typeof record.stdout!=='string'||typeof record.stderr!=='string'||
    Buffer.byteLength(record.stdout,'utf8')+Buffer.byteLength(record.stderr,'utf8')>task.maxOutputBytes||
    record.stdoutSha256!==digest(record.stdout)||record.stderrSha256!==digest(record.stderr)||
    typeof record.startedAt!=='string'||!Number.isFinite(Date.parse(record.startedAt))||
    typeof record.endedAt!=='string'||!Number.isFinite(Date.parse(record.endedAt))||
    Date.parse(record.endedAt)<Date.parse(record.startedAt))throw Error('SANDBOX_RECEIPT_CONTRACT_VIOLATION');
  return record;
}
