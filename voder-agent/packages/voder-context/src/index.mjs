/**
 * Trusted, evidence-aware contextual policies. No ambient tool grants, memory, or extra model calls.
 * Preflight and delivery operate as host lifecycle gates; next/repair are model-facing overlays.
 */
import { readFile, lstat, realpath, open } from 'node:fs/promises';
import { readFileSync, constants } from 'node:fs';
import { createHash } from 'node:crypto';
import { join, relative, isAbsolute, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { TextDecoder } from 'node:util';
import { VoderError } from '../../voder-agent-core/src/contracts.mjs';

const PACKAGE_ROOT = fileURLToPath(new URL('../', import.meta.url));
const SKILLS_ROOT = join(PACKAGE_ROOT, 'skills');
const POLICY_ROOT = join(PACKAGE_ROOT, 'policies');
const NAME_RE = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
// Host-owned manifest: installing a skill requires an explicit manifest hash pin, not workspace discovery.
function installedManifest(){
 let doc;
 try{doc=JSON.parse(readFileSync(join(SKILLS_ROOT,'manifest.json'),'utf8'));}
 catch{fail('INVALID_SKILL_MANIFEST','Cannot read trusted skills manifest');}
 if(doc?.schema!=='voder.context.skills.v1'||!Array.isArray(doc.skills)||doc.skills.length>32||doc.skills.length<1)
   fail('INVALID_SKILL_MANIFEST','Invalid skills manifest schema or size');
 const names=new Set();
 for(const record of doc.skills){
  if(!record||Object.keys(record).sort().join(',')!=='name,sha256'||typeof record.name!=='string'||!NAME_RE.test(record.name)||record.name.length>64||
   typeof record.sha256!=='string'||!/^[a-f0-9]{64}$/.test(record.sha256)||names.has(record.name))
   fail('INVALID_SKILL_MANIFEST','Invalid, duplicate or unpinned manifest entry');
  names.add(record.name);
 }
 return Object.freeze(doc.skills.map(r=>Object.freeze({...r})));
}
const INSTALLED_MANIFEST=installedManifest();
// Installed, version-pinned policies are host-owned data. A workspace MD cannot replace them.
const POLICY_NAMES = Object.freeze(['golden-rules','next','repair','preflight','delivery','strategy']);
function installedPolicies(){
 let doc;
 try{doc=JSON.parse(readFileSync(join(POLICY_ROOT,'manifest.json'),'utf8'));}
 catch{fail('INVALID_POLICY_MANIFEST','Cannot read policy manifest');}
 if(doc?.schema!=='voder.context.policies.v1'||!Array.isArray(doc.policies)||doc.policies.length!==POLICY_NAMES.length)
  fail('INVALID_POLICY_MANIFEST','Invalid policy manifest');
 const mapped=new Map();
 for(const entry of doc.policies){
  if(!entry || Object.keys(entry).sort().join(',')!=='name,sha256' ||
     !POLICY_NAMES.includes(entry.name) || typeof entry.sha256!=='string' || !/^[a-f0-9]{64}$/.test(entry.sha256) || mapped.has(entry.name))
    fail('INVALID_POLICY_MANIFEST','Malformed policy identity');
  mapped.set(entry.name,entry.sha256);
 }
 if(mapped.size!==POLICY_NAMES.length)fail('INVALID_POLICY_MANIFEST','Missing policy identity');
 return mapped;
}
const INSTALLED_POLICIES=installedPolicies();
const KNOWN_SKILLS=Object.freeze(INSTALLED_MANIFEST.map(x=>x.name));
const STAGES = new Set(['next','repair','preflight','delivery']);
const UTF8 = new TextDecoder('utf-8', {fatal:true});
const MAX_SELECTED = 8;
const MAX_SKILL_BYTES = 4096;
const MAX_POLICY_BYTES = 4096;
const DEFAULT_LIMIT = 8192;
const SHA = b => createHash('sha256').update(b).digest('hex');
const fail = (code, message) => {throw new VoderError(code,message);};

function validName(name){
 if(typeof name!=='string' || name.length>64 || !NAME_RE.test(name)) fail('INVALID_SKILL','Invalid skill name');
 return name;
}
function validBudget(b){
 if(!Number.isSafeInteger(b)||b<512||b>24000) fail('INVALID_CONTEXT_BUDGET','Supplement budget must be 512..24000 bytes');
 return b;
}
async function trustedFile(root, slashPath, limit){
 const base=await realpath(root);
 if(!isAbsolute(base)) fail('UNTRUSTED_RESOURCE','Trusted resource root must be absolute');
 let cursor=base;
 for(const component of slashPath.split('/')){
  if(!component || component==='.' || component==='..' || component.includes('\\') || component.includes('\0')) fail('UNTRUSTED_RESOURCE','Invalid resource path');
  cursor=join(cursor,component);
  const st=await lstat(cursor);
  if(st.isSymbolicLink()) fail('UNTRUSTED_RESOURCE','Symlink resource forbidden');
 }
 const normalized=await realpath(cursor);
 const rel=relative(base,normalized);
 if(rel==='..'||rel.startsWith('..'+sep)||isAbsolute(rel)) fail('UNTRUSTED_RESOURCE','Resource escaped trusted root');
 const handle=await open(normalized, constants.O_RDONLY | (constants.O_NOFOLLOW??0));
 try{
  const st=await handle.stat();
  if(!st.isFile()||st.size<1||st.size>limit) fail('CONTEXT_RESOURCE_LIMIT','Trusted file is empty or too large');
  const data=await handle.readFile();
  if(data.length<1||data.length>limit) fail('CONTEXT_RESOURCE_LIMIT','Resource changed outside size limit');
  if(await realpath(cursor)!==normalized) fail('UNTRUSTED_RESOURCE','Resource changed location during read');
  let text;
  try {text=UTF8.decode(data);}catch{fail('INVALID_CONTEXT_UTF8','Context resource is not UTF-8');}
  return Object.freeze({text,sha256:SHA(data),bytes:data.length});
 }finally{await handle.close();}
}
function parseSkill(file, name){
 const match=/^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]+)$/.exec(file.text);
 if(!match) fail('INVALID_SKILL','SKILL.md requires bounded frontmatter');
 const metadata=new Map();
 for(const line of match[1].split(/\r?\n/)){
  const kv=/^([a-z][a-z-]*):\s*(.*?)\s*$/.exec(line);
  if(!kv||metadata.has(kv[1])||!['name','description','triggers'].includes(kv[1])) fail('INVALID_SKILL','Unknown or duplicate skill metadata');
  metadata.set(kv[1],kv[2]);
 }
 const description=metadata.get('description');
 if(metadata.get('name')!==name||!description||description.length>256) fail('INVALID_SKILL','Invalid skill metadata');
 const triggers=(metadata.get('triggers')??'').split(',').map(s=>s.trim().toLowerCase()).filter(Boolean);
 if(triggers.length>24||triggers.some(t=>t.length>48||t.includes('\\'))) fail('INVALID_SKILL','Invalid skill trigger');
 const body=match[2].trim();
 if(!body)fail('INVALID_SKILL','Skill body is missing');
 return Object.freeze({name,description,triggers,body,sha256:file.sha256});
}
const repairTerms = ['bug','fix','repair','debug','error','exception','traceback','stderr','failure','failed','修复','报错','错误','根因','异常','失败','例外','修正','不具合','バグ','修復','오류','버그','수정'];
const codeTerms = ['code','script','program','test','sandbox','compile','build','execute','python','java','rust','javascript','typescript','go','c++','shell','node','pytest','代码','脚本','沙盒','执行','测试','编译','运行','程序','开发','调试','修订','修改','生成','csharp','dotnet','kotlin','swift','ruby','php','scala','powershell','cobol','代码审查','編譯','コード','修正','오류','수정'];
function hits(text, terms){
 const lower=text.toLowerCase();
 return terms.some(word => /^[a-z0-9+-]+$/.test(word)
  ? new RegExp(`(^|[^a-z0-9])${word.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')}($|[^a-z0-9])`,'u').test(lower)
  : lower.includes(word));
}
function skillHits(task,triggers){return hits(task,triggers);}
function skillsSelection(selectedSkills){
 if(!Array.isArray(selectedSkills)||selectedSkills.length>MAX_SELECTED) fail('INVALID_SKILL_SET','At most 8 installed skills may be selected');
 const selected=selectedSkills.map(validName);
 if(new Set(selected).size!==selected.length) fail('INVALID_SKILL_SET','Duplicate skills');
 for(const name of selected) if(!KNOWN_SKILLS.includes(name)) fail('UNKNOWN_SKILL',`Skill is not host-installed: ${name}`);
 return selected.sort();
}

/**
 * Extract only explicit input attachment markers and @file/path references.
 * No speculative guessed files, no filenames inside ordinary prose, no path auto-creation.
 * A user-supplied requiredFiles array remains authoritative and is checked by kernel preflight.
 */
export function inferExplicitRequiredFiles(goal){
 if(typeof goal!=='string') fail('INVALID_TASK','Task must be a string');
 if(goal.length>100000) fail('CONTEXT_RESOURCE_LIMIT','Task exceeds preflight scan limit');
 const candidates=[];
 const rx= /<<File name="([^"]{1,240})">>|(?:^|[\s(])@([A-Za-z0-9_.\-/]+(?:\.[A-Za-z0-9]{1,12}))(?![A-Za-z0-9])/gm;
 for(const m of goal.matchAll(rx)){
  const name=(m[1]??m[2]).trim();
  if(!name||name.includes('\0')||name.includes('\\')||name.startsWith('/')||name==='.'||name==='..'||name.split('/').includes('..')||name.split('/').includes('.')) continue;
  if(!candidates.includes(name)) candidates.push(name);
  if(candidates.length>24) fail('CONTEXT_RESOURCE_LIMIT','Too many explicit input files');
 }
 return Object.freeze(candidates);
}

/** Frozen, versionable trusted context composition; no ambient resource discovery. */
export function createVoderContext({selectedSkills=[],autoSelect=true,maxSupplementBytes=DEFAULT_LIMIT}={}){
 const explicit=skillsSelection(selectedSkills);
 if(typeof autoSelect!=='boolean') fail('INVALID_CONTEXT_POLICY','autoSelect must be boolean');
 const budget=validBudget(maxSupplementBytes);
 // The manifest lists all available skills, but load and validate only selected skill bodies.
 // Never load workspace or user-provided SKILL.md into the system prompt.
 async function policy(name){
  const file=await trustedFile(POLICY_ROOT,`${name}.md`,MAX_POLICY_BYTES);
  if(file.sha256!==INSTALLED_POLICIES.get(name))fail('POLICY_INTEGRITY_MISMATCH',`Installed policy changed: ${name}`);
  return file;
 }
 async function skill(name){
  const pin=INSTALLED_MANIFEST.find(x=>x.name===name);
  const file=await trustedFile(SKILLS_ROOT,`${name}/SKILL.md`,MAX_SKILL_BYTES);
  if(file.sha256!==pin.sha256)fail('SKILL_INTEGRITY_MISMATCH',`Installed skill changed: ${name}`);
  return parseSkill(file,name);
 }
 async function plan({stage='next',task,signals}={}){
  if(!STAGES.has(stage)) fail('INVALID_STAGE','Unsupported context stage');
  if(typeof task!=='string'||!task.trim()||task.length>100000) fail('INVALID_TASK','Task is empty or excessive');
  if(signals!==undefined && (signals===null||typeof signals!=='object'||Array.isArray(signals) ||
      Object.keys(signals).some(k=>!['diagnostic','requiresExecution','quantitative','hasMissingInput'].includes(k)) ||
      Object.values(signals).some(x=>typeof x!=='boolean')))
    fail('INVALID_CONTEXT_SIGNALS','Only known host-owned boolean signals are permitted');
  // Signals come from the trusted host, not from model output or workspace text.
  const hints=signals??{};
  const isRepair=stage==='repair'||(stage==='next'&&(hints.diagnostic===true||hits(task,repairTerms)));
  const isCode=hints.requiresExecution===true||hits(task,codeTerms);
  const selected=[];
  for(const name of KNOWN_SKILLS){
   // Explicitly configured skills remain mandatory; automatic matches are opportunistic.
   let should=explicit.includes(name);
   if(!should && autoSelect){
    if(name==='root-cause-analysis') should=isRepair;
    else if(name==='sandbox-verification') should=isRepair||isCode;
    else if(name==='scientific-validation') should=hints.quantitative===true||hits(task,['math','mathematics','physics','numerical','simulation','scientific','statistics','quantitative','数学','物理','数值','统计','仿真','模拟','數學','物理学']);
    else if(name==='missing-input-audit') should=hints.hasMissingInput===true||hits(task,['missing','required','attachment','附件','缺失','上传','不足','缺少']);
   }
   if(should)selected.push(name);
  }
  const effectiveStage=stage==='next'&&isRepair?'repair':stage;
  // The canonical golden-rules and next overlay remain first in prompt.mjs, not duplicated here.
  // Verify their published digests as well, preventing split-brain prompt identity.
  const core=await policy('golden-rules');
  const next=await policy('next');
  const strategy=await policy('strategy');
  const policyHashes={golden:core.sha256,next:next.sha256,strategy:strategy.sha256};
  const pieces=['[VODER_CONTEXT_V2]',`[POLICY:strategy sha256=${strategy.sha256}]\n${strategy.text.trim()}\n[/POLICY]`];
  if(stage==='preflight'||stage==='delivery'){
   const stageFile=await policy(stage);
   policyHashes[stage]=stageFile.sha256;
   pieces.push(`[POLICY:${stage} sha256=${stageFile.sha256}]\n${stageFile.text.trim()}\n[/POLICY]`);
  }
  if(isRepair){
   const repair=await policy('repair');
   policyHashes.repair=repair.sha256;
   pieces.push(`[POLICY:repair sha256=${repair.sha256}]\n${repair.text.trim()}\n[/POLICY]`);
  }
  const chosen=[];
  const omitted=[];
  for(const name of [...explicit.filter(n=>selected.includes(n)),...selected.filter(n=>!explicit.includes(n))]){
   const record=await skill(name);
   const candidate=`[SKILL:${name} sha256=${record.sha256}]\n${record.body}\n[/SKILL]`;
   const trial=[...pieces,candidate,'[/VODER_CONTEXT_V2]'].join('\n\n');
   if(Buffer.byteLength(trial,'utf8')>budget){
    if(explicit.includes(name))fail('CONTEXT_OVERFLOW',`Explicitly required skill exceeds budget: ${name}`);
    omitted.push(name);continue;
   }
   pieces.push(candidate);chosen.push(name);
  }
  pieces.push('[/VODER_CONTEXT_V2]');
  const supplement=pieces.join('\n\n');
  const bytes=Buffer.byteLength(supplement,'utf8');
  if(bytes>budget) fail('CONTEXT_OVERFLOW','Mandatory context policies exceed bounded budget; not truncated');
  return Object.freeze({stage:effectiveStage,selectedSkills:Object.freeze([...chosen]),omittedSkills:Object.freeze([...omitted]),
    bytes,estimatedTokensUpperBound:bytes,sha256:SHA(supplement),policyHashes:Object.freeze(policyHashes),supplement});
 }
 // Lifecycle control is host-only: before/after a task, inspect stage policy without
 // issuing another model call, granting tools, or claiming stronger native verification.
 async function preparePhase({stage,task,signals}={}){
  if(stage!=='preflight'&&stage!=='delivery')fail('INVALID_STAGE','Lifecycle phase must be preflight or delivery');
  const p=await plan({stage,task,signals});
  return Object.freeze({stage:p.stage,sha256:p.sha256,policyHashes:p.policyHashes,
     selectedSkills:p.selectedSkills,omittedSkills:p.omittedSkills,bytes:p.bytes});
 }
 async function reviewDelivery({task,status,nativeCommit,sandboxRequired=false,prepared}={}){
  const policyIdentity=prepared??await preparePhase({stage:'delivery',task});
  if(policyIdentity.stage!=='delivery' || !/^[a-f0-9]{64}$/.test(policyIdentity.sha256??''))fail('INVALID_DELIVERY_AUDIT','Invalid prepared delivery policy');
  const committed=status==='VERIFIED'&&nativeCommit?.status==='COMMITTED';
  // Native commit is the only grant. This observer can never make a blocked task verified.
  if(status==='VERIFIED'&&!committed)fail('DELIVERY_EVIDENCE_MISSING','Cannot close without native COMMITTED receipt');
  return Object.freeze({...policyIdentity,outcome:committed?'NATIVE_VERIFIED':status==='DRAFT_UNVERIFIED'?'UNVERIFIED_DRAFT':
     status==='WAITING_FOR_INPUT'?'MISSING_INPUT':'NOT_VERIFIED',sandboxRequired:!!sandboxRequired});
 }
 return Object.freeze({plan,preparePhase,reviewDelivery,provider:async input=>(await plan(input)).supplement});
}
export function createVoderContextProvider(options){
 const manager=createVoderContext(options);
 const provider=async input=>manager.provider(input);
 Object.defineProperties(provider,{
  preparePhase:{value:manager.preparePhase,enumerable:false},
  reviewDelivery:{value:manager.reviewDelivery,enumerable:false}
 });
 return Object.freeze(provider);
}

/** Canonical bundle read at startup by prompt.mjs, not on every tool step. */
export function readBundledPolicySync(filename){
 if(!['golden-rules.md','next.md','repair.md','preflight.md','delivery.md','strategy.md'].includes(filename)) fail('INVALID_POLICY','Unknown policy name');
 return readFileSync(join(POLICY_ROOT,filename),'utf8');
}
