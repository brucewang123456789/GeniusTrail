/** Opt-in installed provider catalogue. The operator, not the model/workspace, grants execution. */
import {readFile,lstat,realpath} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {join,resolve,basename} from 'node:path';
import {pathToFileURL} from 'node:url';
import {PluginRegistry} from './plugin.ts';

const ID=/^[a-z][a-z0-9-]{1,39}$/;
const HEX=/^[0-9a-f]{64}$/;
const ENTRY=/^[a-z][a-z0-9-]{1,39}-[0-9a-f]{64}\.mjs$/;
const SCHEMA='voder.sandbox.plugins.v1';
export type InstalledEntry={id:string;version:string;entry:string;sha256:string};
export async function catalogue(root:string):Promise<InstalledEntry[]> {
  if(typeof root!=='string'||!root.startsWith('/')||resolve(root)!==root)throw Error('PLUGIN_ROOT_NOT_ABSOLUTE_NORMALIZED');
  const st=await lstat(root);
  if(!st.isDirectory()||st.isSymbolicLink()||(st.mode&0o022)!==0||await realpath(root)!==root)throw Error('UNTRUSTED_PLUGIN_ROOT');
  const location=join(root,'installed.json');
  let raw:string;
  try{const s=await lstat(location);if(!s.isFile()||s.isSymbolicLink()||s.size>65536)throw Error('PLUGIN_CATALOGUE_UNSAFE');raw=await readFile(location,'utf8');}
  catch(e){if((e as NodeJS.ErrnoException).code==='ENOENT')return [];throw e;}
  const doc:unknown=JSON.parse(raw);
  if(!doc||typeof doc!=='object'||Array.isArray(doc))throw Error('PLUGIN_CATALOGUE_INVALID');
  const value=doc as {schema?:unknown;providers?:unknown};
  if(value.schema!==SCHEMA||!Array.isArray(value.providers)||value.providers.length>32)throw Error('PLUGIN_CATALOGUE_INVALID');
  const names=new Set<string>();
  return value.providers.map((v:unknown)=>{
    if(!v||typeof v!=='object'||Array.isArray(v))throw Error('PLUGIN_CATALOGUE_INVALID');
    const o=v as Record<string,unknown>;
    if(typeof o.id!=='string'||!ID.test(o.id)||o.id==='modal'||names.has(o.id)||typeof o.version!=='string'||o.version.length>64||
       typeof o.sha256!=='string'||!HEX.test(o.sha256)||typeof o.entry!=='string'||!ENTRY.test(o.entry)||
       o.entry!==`${o.id}-${o.sha256}.mjs`||Object.keys(o).sort().join(',')!=='entry,id,sha256,version')throw Error('PLUGIN_CATALOGUE_INVALID');
    names.add(o.id);return o as InstalledEntry;
  });
}
async function auditedEntrypoint(root:string,entry:InstalledEntry):Promise<string>{
  const location=join(root,entry.entry);
  if(basename(location)!==entry.entry)throw Error('PLUGIN_PATH_ESCAPE');
  const st=await lstat(location);
  if(!st.isFile()||st.isSymbolicLink()||st.nlink!==1||st.size>2*1024*1024||(st.mode&0o022)!==0)throw Error('PLUGIN_CODE_UNSAFE');
  const bytes=await readFile(location);
  if(createHash('sha256').update(bytes).digest('hex')!==entry.sha256)throw Error('PLUGIN_DIGEST_MISMATCH');
  const after=await lstat(location);
  if(after.ino!==st.ino||after.dev!==st.dev||after.size!==st.size||after.mtimeMs!==st.mtimeMs)throw Error('PLUGIN_CHANGED');
  return location;
}
/** Installed plugins are lazy: a third-party plugin is not imported unless selected. */
export async function registerInstalledPlugins(registry:PluginRegistry,root:string|undefined=process.env.VODER_SANDBOX_PLUGIN_DIR):Promise<void>{
  if(!root)return;
  for(const item of await catalogue(root)){
    registry.register(item.id,async()=>{
      const file=await auditedEntrypoint(root,item);
      // Administrator-installed JavaScript is trusted code. Never source from a user workspace.
      const mod=await import(pathToFileURL(file).href);
      if(typeof mod.createSandboxPlugin!=='function')throw Error('SANDBOX_PLUGIN_EXPORT_MISSING');
      const instance=await mod.createSandboxPlugin();
      if(instance?.id!==item.id||instance?.version!==item.version)throw Error('PLUGIN_IDENTITY_MISMATCH');
      return instance;
    });
  }
}
