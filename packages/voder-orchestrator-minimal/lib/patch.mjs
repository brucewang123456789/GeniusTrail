import {lstat,readFile,open,mkdir,unlink,rename} from 'node:fs/promises';
import {join,dirname} from 'node:path';
import {sha,insist,relPath} from './safety.mjs';
const LIMIT=1024*1024;
export function parseProposal(text,allowed){insist(typeof text==='string'&&Buffer.byteLength(text)<131072,'MODEL_PROPOSAL_TOO_LARGE');let data;try{data=JSON.parse(text.trim());}catch{throw new Error('MODEL_MUST_RETURN_RAW_JSON');}
 insist(data&&typeof data==='object'&&!Array.isArray(data)&&data.version===1&&Array.isArray(data.patches)&&data.patches.length>0&&data.patches.length<=16,'INVALID_PATCH_SCHEMA');
 insist(Object.keys(data).every(k=>['version','patches','rationale'].includes(k)),'PATCH_UNKNOWN_FIELDS');
 const paths=new Set();for(const p of data.patches){insist(p&&typeof p==='object'&&!Array.isArray(p)&&Object.keys(p).every(k=>['path','action','beforeSha256','content'].includes(k)),'PATCH_UNKNOWN_FIELDS');relPath(p.path);insist(allowed.has(p.path)&&!paths.has(p.path),'PATCH_OUTSIDE_ALLOWED_SCOPE');paths.add(p.path);insist(['replace','create','delete'].includes(p.action),'INVALID_PATCH_ACTION');insist(['replace','delete'].includes(p.action)?/^[0-9a-f]{64}$/.test(p.beforeSha256):p.beforeSha256===undefined,'INVALID_PATCH_PREIMAGE');insist(p.action==='delete'?p.content===undefined:typeof p.content==='string'&&Buffer.byteLength(p.content,'utf8')<=LIMIT,'INVALID_PATCH_CONTENT');}
 return data;
}
async function inspect(p){try{const st=await lstat(p);insist(st.isFile()&&!st.isSymbolicLink()&&st.nlink===1&&st.size<=LIMIT,'UNSAFE_PATCH_TARGET');return readFile(p);}catch(e){if(e.code==='ENOENT')return null;throw e;}}
export async function applyPatches(root,proposal){const applied=[];
 try{for(const p of proposal.patches){const full=join(root,p.path);const before=await inspect(full);insist(p.action==='create'?before===null:before!==null,'PATCH_PREIMAGE_EXISTENCE');if(p.beforeSha256)insist(sha(before)===p.beforeSha256,'PATCH_PREIMAGE_CHANGED');const parent=dirname(full);let ps=await lstat(parent);insist(ps.isDirectory()&&!ps.isSymbolicLink(),'PATCH_PARENT_UNSAFE');
  if(p.action==='delete')await unlink(full);
  else {const tmp=join(parent,`.voder-stage-${process.pid}-${Math.random().toString(16).slice(2)}`);let fd;try{fd=await open(tmp,'wx',0o600);await fd.writeFile(p.content);await fd.sync();await fd.close();fd=null;await rename(tmp,full);}finally{if(fd)await fd.close().catch(()=>{});await unlink(tmp).catch(()=>{});}}
  applied.push({path:p.path,before,after:p.action==='delete'?null:Buffer.from(p.content)});
 }return applied;
 }catch(e){await undo(root,applied).catch(()=>{});throw e;}
}
export async function undo(root,applied){for(const p of [...applied].reverse()){const dst=join(root,p.path);const now=await inspect(dst);insist((now===null?null:sha(now))===(p.after===null?null:sha(p.after)),'ROLLBACK_PREIMAGE_CHANGED');if(p.before===null){if(now!==null)await unlink(dst);}else{const tmp=join(dirname(dst),`.voder-rollback-${process.pid}-${Math.random().toString(16).slice(2)}`);let fd;try{fd=await open(tmp,'wx',0o600);await fd.writeFile(p.before);await fd.sync();await fd.close();fd=null;await rename(tmp,dst);}finally{if(fd)await fd.close().catch(()=>{});await unlink(tmp).catch(()=>{});}}}}
