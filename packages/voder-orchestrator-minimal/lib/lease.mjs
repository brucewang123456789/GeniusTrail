import {open,readFile,unlink,lstat} from 'node:fs/promises';
import {join} from 'node:path';
import {randomBytes} from 'node:crypto';
import {insist} from './safety.mjs';

// Deliberately fail closed on a stale lease; only a trusted operator can
// reconcile a crashed run after verifying journal and staged artifacts.
export async function withRunLease(stateRoot,work){
 const name=join(stateRoot,'.voder-orchestrator.lock');
 const nonce=randomBytes(24).toString('hex');
 let file;
 try{file=await open(name,'wx',0o600);}catch(e){
  if(e?.code==='EEXIST')throw Error('RECOVERY_REQUIRED_EXISTING_RUN_LEASE');
  throw e;
 }
 let acquired=false;
 try{
  await file.writeFile(JSON.stringify({version:1,nonce,pid:process.pid,created:new Date().toISOString()})+'\n');
  await file.sync();await file.close();file=null;acquired=true;
  return await work();
 }finally{
  if(file)await file.close().catch(()=>{});
  if(acquired){
   const st=await lstat(name).catch(()=>null);
   insist(st?.isFile()&&!st.isSymbolicLink(),'RUN_LEASE_UNCERTAIN');
   const current=JSON.parse(await readFile(name,'utf8'));
   insist(current.nonce===nonce,'RUN_LEASE_TAMPERED');
   await unlink(name);
  }
 }
}
