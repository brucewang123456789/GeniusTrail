#!/usr/bin/env node
/** Trusted operator CLI. A third-party plugin is executable host code: install only audited files. */
import {mkdir,open,lstat,readFile,writeFile,rename,rm,realpath} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {join,resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {catalogue} from './installed.ts';
const ID=/^[a-z][a-z0-9-]{1,39}$/;
const SHA=/^[a-f0-9]{64}$/;
function fail(s){throw Error(s);}
function argsOf(argv){
  const [command,...rest]=argv;const args={};
  for(let i=0;i<rest.length;i+=2){const key=rest[i];if(!key?.startsWith('--')||i+1===rest.length||rest[i+1].startsWith('--')||Object.hasOwn(args,key))fail('INVALID_ARGUMENTS');args[key]=rest[i+1];}
  return {command,args};
}
async function main(){
  const {command,args}=argsOf(process.argv.slice(2));
  if(!['install','remove','list','verify'].includes(command))fail('USAGE: [install|remove|list|verify] --root ABSOLUTE ...');
  const root=args['--root']??process.env.VODER_SANDBOX_PLUGIN_DIR;
  if(!root?.startsWith('/')||resolve(root)!==root)fail('PLUGIN_ROOT_NOT_ABSOLUTE_NORMALIZED');
  const creates=command==='install';
  if(creates)await mkdir(root,{recursive:true,mode:0o700});
  const stat=await lstat(root);
  if(!stat.isDirectory()||stat.isSymbolicLink()||(stat.mode&0o077)!==0||await realpath(root)!==root)fail('PLUGIN_ROOT_NOT_PRIVATE');
  const lock=join(root,'.install.lock');
  let handle;
  if(command==='install'||command==='remove')handle=await open(lock,'wx',0o600);
  try{
    const existing=await catalogue(root);
    if(command==='list'){
      for(const item of existing)process.stdout.write(`${item.id}\t${item.version}\t${item.sha256}\n`);
      return;
    }
    if(command==='verify'){
      for(const item of existing){const p=join(root,item.entry);const s=await lstat(p);
        if(!s.isFile()||s.isSymbolicLink()||s.nlink!==1||(s.mode&0o022)!==0||
          createHash('sha256').update(await readFile(p)).digest('hex')!==item.sha256)fail('PLUGIN_VERIFY_FAILED:'+item.id);
      }
      process.stdout.write('PLUGIN_CATALOGUE_GATE=PASS\n');return;
    }
    const id=args['--id'];
    if(!id||!ID.test(id)||id==='modal')fail('INVALID_OR_RESERVED_PLUGIN_ID');
    const old=existing.find(p=>p.id===id);
    if(command==='remove'){
      if(!old)fail('PLUGIN_NOT_INSTALLED');
      const next=existing.filter(p=>p.id!==id);
      await saveCatalogue(root,next);
      // Do not delete plugin executable after withdrawal: concurrent historical processes may depend on it.
      process.stdout.write(`PLUGIN_UNREGISTERED=${id}\n`);return;
    }
    if(old)fail('PLUGIN_ALREADY_INSTALLED');
    const input=args['--file'],pin=args['--sha256'],version=args['--version'];
    if(!input?.startsWith('/')||resolve(input)!==input||!pin||!SHA.test(pin)||!version||version.length>64||!/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$/.test(version))fail('INVALID_INSTALL_PARAMETERS');
    if(existing.length>=32)fail('PLUGIN_CAPACITY_EXCEEDED');
    const s=await lstat(input);
    if(!s.isFile()||s.isSymbolicLink()||s.nlink!==1||s.size>2*1024*1024||(s.mode&0o022)!==0)fail('UNSAFE_PLUGIN_INPUT');
    const bytes=await readFile(input);const sha256=createHash('sha256').update(bytes).digest('hex');
    if(sha256!==pin)fail('PLUGIN_SOURCE_DIGEST_MISMATCH');
    const entry=`${id}-${sha256}.mjs`,dest=join(root,entry);
    // No untrusted package manager side effects or install hooks.
    const created=await open(dest,'wx',0o600);
    try{await created.writeFile(bytes);}finally{await created.close();}
    try{await saveCatalogue(root,[...existing,{id,version,entry,sha256}]);}
    catch(e){await rm(dest,{force:true});throw e;}
    process.stdout.write(`SANDBOX_PLUGIN_INSTALLED=${id}\nPLUGIN_SHA256=${pin}\n`);
  }finally{if(handle){await handle.close();await rm(lock,{force:true});}}
}
async function saveCatalogue(root,providers){
  const temp=join(root,`.installed-${process.pid}-${Date.now()}.tmp`);
  const h=await open(temp,'wx',0o600);
  try{await h.writeFile(JSON.stringify({schema:'voder.sandbox.plugins.v1',providers})+'\n');}
  finally{await h.close();}
  try{await rename(temp,join(root,'installed.json'));}catch(e){await rm(temp,{force:true});throw e;}
}
if(import.meta.url===pathToFileURL(process.argv[1]??'').href){main().catch(e=>{process.stderr.write(`PLUGIN_MANAGER_BLOCKED=${String(e?.message??e).slice(0,180)}\n`);process.exitCode=21;});}
