#!/usr/bin/env node
/** Production entrypoint: invoked only by the authorized Rust ToolExecutor backend. */
import {readFile,writeFile,rename,chmod,lstat} from 'node:fs/promises';
import {builtinPlugins,registerInstalledPlugins} from './registry.ts';
import {assertProviderReceipt} from './receipt-gate.ts';
import {validateTask} from './contract.ts';
import {receiptMetadata,type SandboxTask} from './contract.ts';
async function main():Promise<void>{
  const paths=process.argv.slice(2);
  if(paths.length!==4 || paths.some(path=>!path.startsWith('/')) || new Set(paths).size!==4){
    throw new Error('USAGE_ABSOLUTE_DISTINCT_REQUEST_RECEIPT_AND_OUTPUT_PATHS');
  }
  const [requestPath,receiptPath,stdoutPath,stderrPath]=paths as [string,string,string,string];
  const stat=await lstat(requestPath);
  if(!stat.isFile()||stat.isSymbolicLink()||stat.size>128*1024)throw new Error('UNSAFE_REQUEST_FILE');
  const task:SandboxTask=validateTask(JSON.parse(await readFile(requestPath,'utf8')));
  const provider=process.env.VODER_SANDBOX_BACKEND??'modal';
  if(provider==='none')throw new Error('SANDBOX_DISABLED_NO_EXECUTION');
  const registry=builtinPlugins();
  if(provider!=='modal')await registerInstalledPlugins(registry);
  const plugin=await registry.resolve(provider);
  const result=assertProviderReceipt(await plugin.execute(task),task,provider);
  // Write raw execution evidence to dedicated files. Never multiplex Node's
  // runtime warnings or SDK diagnostic stderr into remote process stderr.
  // Exclusive create protects against accidental reuse of a capture directory.
  await writeFile(stdoutPath,result.stdout,{mode:0o600,flag:'wx'});
  await writeFile(stderrPath,result.stderr,{mode:0o600,flag:'wx'});
  // Receipt must be atomically written and never contain credentials.
  const tmp=`${receiptPath}.tmp-${process.pid}`;
  await writeFile(tmp,JSON.stringify(receiptMetadata(result))+'\n',{mode:0o600,flag:'wx'});
  await chmod(tmp,0o600);
  await rename(tmp,receiptPath);
  // Remote execution status is in the receipt, not the local transport exit.
  // Local stdout/stderr are reserved for transport diagnostics only.
  // A successful transport with a failing test still needs a valid receipt.
  process.exitCode=0;
}
main().catch(error=>{
  const msg=error instanceof Error?error.message:String(error);
  const safe=msg.replaceAll(process.env.MODAL_TOKEN_SECRET??'__SECRET_UNSET__','[REDACTED]')
                .replaceAll(process.env.MODAL_TOKEN_ID??'__ID_UNSET__','[REDACTED]').slice(0,300);
  process.stderr.write(`VODER_SANDBOX_BLOCKED=${safe}\n`);
  process.exitCode=70;
});
