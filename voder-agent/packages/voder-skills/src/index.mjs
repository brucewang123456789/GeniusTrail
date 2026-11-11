/** Installed, SHA-pinned SKILL.md material. Never grants tools or system authority. */
import {readFile,writeFile,mkdir,link,lstat,readdir,rm,open} from 'node:fs/promises';
import {constants} from 'node:fs';
import {resolve,join,basename} from 'node:path';
import {createHash,randomUUID} from 'node:crypto';
import {VoderError} from '../../voder-agent-core/src/contracts.mjs';
const SHA=/^[0-9a-f]{64}$/;const NAME=/^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$/;
const hash=x=>createHash('sha256').update(x).digest('hex');
const fail=(code,msg)=>{throw new VoderError(code,msg);};
const MAX_SKILL=16*1024;const MAX_TOTAL=24*1024;
function normalize(file,name){
 if(typeof file!=='string'||Buffer.byteLength(file,'utf8')>MAX_SKILL||file.includes('\0')||!file.trim())fail('INVALID_SKILL','Invalid skill contents');
 const m=file.match(/^---\s*\n([\s\S]*?)\n---\s*\n/);
 if(!m)fail('INVALID_SKILL','SKILL.md must include a bounded name and description frontmatter');
 const fields={};for(const line of m[1].split('\n')){const pair=line.match(/^([a-z_]+):\s*(.*?)\s*$/);if(!pair||Object.hasOwn(fields,pair[1]))fail('INVALID_SKILL','Unsupported frontmatter');fields[pair[1]]=pair[2].replace(/^['"]|['"]$/g,'');}
 if(Object.keys(fields).some(k=>!['name','description'].includes(k))||fields.name!==name||!fields.description||fields.description.length>240||!NAME.test(fields.name))
   fail('INVALID_SKILL','Skill metadata conflicts with pinned identity');
 return {name,description:fields.description,body:file};
}
async function regular(path){const s=await lstat(path);if(!s.isFile()||s.isSymbolicLink()||s.size>MAX_SKILL)fail('UNSAFE_SKILL','Skill must be a bounded regular file');
 const h=await open(path,constants.O_RDONLY|constants.O_NOFOLLOW);try{return await h.readFile({encoding:'utf8'});}finally{await h.close();}}
export class InstalledSkills {
 constructor({root}={}){if(typeof root!=='string'||!root.startsWith('/'))fail('INVALID_SKILL_ROOT','Absolute administrator-selected skill storage required');this.root=resolve(root);}
 async install({source,sha256}={}){
  if(typeof source!=='string'||!source.startsWith('/')||basename(source)!=='SKILL.md'||!SHA.test(sha256??''))fail('INVALID_SKILL_SOURCE','Explicit absolute SKILL.md and SHA-256 pin required');
  await mkdir(this.root,{recursive:true,mode:0o700});
  const rootstat=await lstat(this.root);if(rootstat.isSymbolicLink()||!rootstat.isDirectory())fail('UNSAFE_SKILL_ROOT','Unsafe skill storage');
  const body=await regular(source);if(hash(body)!==sha256)fail('SKILL_DIGEST_MISMATCH','Source differs from install pin');
  const name=basename(resolve(source,'..'));normalize(body,name);
  if(!NAME.test(name)||name.length>64)fail('INVALID_SKILL','Invalid package name');
  const dest=join(this.root,`${name}.json`);try{await lstat(dest);fail('DUPLICATE_SKILL','Skill is already installed');}catch(e){if(e.code!=='ENOENT')throw e;}
  const record=JSON.stringify({schema:'voder.skill.pin.v1',name,sha256,body});
  const tmp=join(this.root,`.install-${randomUUID()}`);
  try{await writeFile(tmp,record,{flag:'wx',mode:0o600});await link(tmp,dest);}finally{await rm(tmp,{force:true});}
  return {name,sha256};
 }
 async list(){
  try{const r=await lstat(this.root);if(!r.isDirectory()||r.isSymbolicLink())fail('UNSAFE_SKILL_ROOT','Unsafe installed skill root');}
  catch(e){if(e.code==='ENOENT')return [];throw e;}
  let names;try{names=await readdir(this.root);}catch(e){if(e.code==='ENOENT')return [];throw e;}
  if(names.length>128)fail('SKILL_STORE_TOO_LARGE','Installed skill registry exceeds limit');
  const result=[];
  for(const file of names){if(!/^[a-z][a-z0-9-]*\.json$/.test(file))continue;
   const path=join(this.root,file);const s=await lstat(path);if(!s.isFile()||s.isSymbolicLink()||s.size>MAX_SKILL+500)fail('UNSAFE_SKILL','Unsafe installed skill entry');
   let doc;try{doc=JSON.parse(await readFile(path,'utf8'));}catch{fail('INVALID_SKILL','Corrupt installed skill record');}
   if(doc?.schema!=='voder.skill.pin.v1'||doc.name!==file.slice(0,-5)||!SHA.test(doc.sha256??'')||hash(doc.body)!==doc.sha256)fail('SKILL_DIGEST_MISMATCH','Installed skill record changed');
   const item=normalize(doc.body,doc.name);result.push({...item,sha256:doc.sha256});
  }
  return result.sort((a,b)=>a.name.localeCompare(b.name));
 }
 async select({task,explicit=[],maxBytes=MAX_TOTAL}={}){
  if(typeof task!=='string'||task.length>100000||!Array.isArray(explicit)||explicit.length>24||
     explicit.some(n=>typeof n!=='string'||!NAME.test(n))||!Number.isInteger(maxBytes)||maxBytes<1024||maxBytes>65536)
    fail('INVALID_SKILL_SELECTION','Invalid task or skill budget');
  const all=await this.list();const found=new Map(all.map(x=>[x.name,x]));
  if(explicit.some(x=>!found.has(x)))fail('MISSING_EXPLICIT_SKILL','Explicit skill not installed');
  const lower=task.toLocaleLowerCase();const selected=[];const omitted=[];let bytes=0;
  const candidates=[...new Set(explicit),...all.filter(x=>!explicit.includes(x.name)&&x.name.split('-').some(w=>w.length>3&&lower.includes(w))).map(x=>x.name)];
  for(const name of candidates){const s=found.get(name);const size=Buffer.byteLength(s.body,'utf8');
   if(bytes+size>maxBytes){if(explicit.includes(name))fail('SKILL_BUDGET_EXHAUSTED','Explicit skill must fit whole or block');omitted.push(name);continue;}
   bytes+=size;selected.push(s);
  }
  return Object.freeze({selected:Object.freeze(selected.map(x=>Object.freeze({...x}))),omitted:Object.freeze(omitted),bytes});
 }
 async remove(name){if(!NAME.test(name))fail('INVALID_SKILL','Bad installed skill name');const existing=(await this.list()).some(x=>x.name===name);
  if(!existing)fail('MISSING_SKILL','Installed skill absent');await rm(join(this.root,`${name}.json`));}
}
/** Skills are task-scoped untrusted materials, not an elevated system prompt. */
export function renderSelectedSkills(selection){return selection.selected.map(s=>
 `[UNTRUSTED_INSTALLED_SKILL name=${s.name} sha256=${s.sha256}]\n${s.body}\n[/UNTRUSTED_INSTALLED_SKILL]`).join('\n\n');}
