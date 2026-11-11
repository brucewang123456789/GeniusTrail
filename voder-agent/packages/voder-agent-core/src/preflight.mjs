import { lstat, realpath } from 'node:fs/promises';
import { resolve, relative, isAbsolute, sep } from 'node:path';
import { VoderError } from './contracts.mjs';
export async function preflightFiles(cwd, required=[]) {
 const base=await realpath(cwd); const missing=[]; const found=[];
 for(const raw of required) {
   if(typeof raw!=='string'|| !raw || raw.includes('\0')||isAbsolute(raw)||raw.split(/[\\/]/).includes('..')) throw new VoderError('INVALID_PATH','Unsafe required path');
   const full=resolve(base,raw);const rel=relative(base,full);
   if(rel.startsWith('..'+sep)||rel==='..'||isAbsolute(rel)) throw new VoderError('INVALID_PATH','Escaping workspace');
   try { const meta=await lstat(full); if(meta.isSymbolicLink()) throw new VoderError('UNTRUSTED_PATH','Required file symlink');
      if(!meta.isFile()) throw new VoderError('INVALID_PATH','Required entry not regular file');
      const real=await realpath(full); const re=relative(base,real);
      if(re==='..'||re.startsWith('..'+sep)||isAbsolute(re)) throw new VoderError('UNTRUSTED_PATH','Symlink parent path escape');
      found.push({path:raw,realpath:real});
   } catch(e) { if(e.code==='ENOENT') missing.push({path:raw,reason:'NOT_FOUND'}); else throw e; }
 }
 return Object.freeze({status:missing.length?'WAITING_FOR_INPUT':'READY',found,missing});
}
